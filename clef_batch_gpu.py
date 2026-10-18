#!/usr/bin/env python3
"""Run recorded Jev requests through local Clef-flash and append resumable answers.

This entry point is intended for the Windows A770 environment. It only resolves the
model snapshot from the local Hugging Face cache; it never downloads model files.
"""
import argparse
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

from jev_stub import request_hash

MODEL_ID = 'Cloudflare/clef-flash'


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSONL: {exc}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: each JSONL row must be an object')
            if not row.get('hash') and not row.get('request_hash'):
                row['hash'] = request_hash(row['state'], row['questions'])
            rows.append(row)
    return rows


def load_answered(path):
    answered = {}
    if not os.path.exists(path):
        return answered
    for line_no, row in enumerate(read_jsonl(path), 1):
        digest = row.get('hash') or row.get('request_hash')
        values = row.get('answers')
        if not digest or not isinstance(values, dict):
            raise ValueError(f'{path}:{line_no}: expected hash and answers object')
        answered[digest] = values
    return answered


def invoke_compatible(fn, available, fallback=(), extra_kwargs=None):
    """Call a documented helper using matching parameter names plus positional fallbacks."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*fallback)
    args, kwargs = [], {}
    fallbacks = iter(fallback)
    has_var_kwargs = False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            continue
        if param.name in available:
            value = available[param.name]
        elif param.default is inspect.Parameter.empty:
            try:
                value = next(fallbacks)
            except StopIteration as exc:
                raise TypeError(f'cannot supply required parameter {param.name!r} to {fn.__name__}') from exc
        else:
            continue
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[param.name] = value
    if has_var_kwargs and extra_kwargs:
        for key, value in extra_kwargs.items():
            kwargs.setdefault(key, value)
    return fn(*args, **kwargs)


def _to_python(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_python(x) for x in value]
    if isinstance(value, list):
        return [_to_python(x) for x in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_record_logits(logits, records):
    """Normalize [batch, questions, options] or equivalent nested output to records."""
    logits = _to_python(logits)
    batch_size = len(records)
    counts = [len(r['questions']) for r in records]
    if isinstance(logits, dict):
        if 'logits' in logits:
            logits = logits['logits']
        else:
            # Question-keyed output is accepted when every question has one option row per record.
            per_record = [[] for _ in records]
            for qi, question in enumerate(records[0]['questions']):
                values = _to_python(logits[question])
                if batch_size == 1 and values and _is_number(values[0]):
                    values = [values]
                for bi in range(batch_size):
                    per_record[bi].append(values[bi])
            return per_record
    if not isinstance(logits, list):
        raise TypeError('model(batch)[0] must return nested option logits')

    # One question, one record may be represented as a bare [false, true] row.
    if batch_size == 1 and counts == [1] and logits and all(_is_number(x) for x in logits):
        return [[logits]]

    # [batch, questions, options], with a one-question record sometimes shortened to [batch, options].
    if len(logits) == batch_size:
        packed = []
        for bi, record_logits in enumerate(logits):
            if counts[bi] == 1 and isinstance(record_logits, list) and record_logits and all(_is_number(x) for x in record_logits):
                record_logits = [record_logits]
            if not isinstance(record_logits, list) or len(record_logits) != counts[bi]:
                packed = []
                break
            packed.append(record_logits)
        if packed:
            return packed

    # [sum(question_count), options], flattened across the batch.
    if len(logits) == sum(counts) and all(isinstance(x, list) and x and all(_is_number(y) for y in x) for x in logits):
        packed, cursor = [], 0
        for count in counts:
            packed.append(logits[cursor:cursor + count])
            cursor += count
        return packed
    raise ValueError('could not map model option logits back to the input records')


def _option_labels(spec, encoded_record, question_name):
    # The release encoder records the actual option ordering here (noul is true, false).
    encoded_questions = getattr(encoded_record, 'questions', None)
    if encoded_questions is None and isinstance(encoded_record, dict):
        encoded_questions = encoded_record.get('questions')
    if encoded_questions:
        for encoded_question in encoded_questions:
            if getattr(encoded_question, 'question_id', None) == question_name:
                option_ids = getattr(encoded_question, 'option_ids', None)
                if option_ids:
                    return list(option_ids)
            if isinstance(encoded_question, dict) and encoded_question.get('question_id') == question_name:
                option_ids = encoded_question.get('option_ids')
                if option_ids:
                    return list(option_ids)
    if isinstance(encoded_record, dict):
        for key in ('question_options', 'option_order', 'option_labels', 'choices', 'options', 'labels'):
            candidate = encoded_record.get(key)
            if isinstance(candidate, dict):
                candidate = candidate.get(question_name)
                if isinstance(candidate, dict):
                    candidate = candidate.get('option_ids', candidate.get('option_order', candidate.get('options')))
            if isinstance(candidate, (list, tuple)):
                labels = [str(x.get('name', x.get('label', x.get('id', '')))) if isinstance(x, dict) else str(x)
                          for x in candidate]
                if labels:
                    return labels
    for candidate in (spec.get('option_order'), spec.get('options'), spec.get('choices'), spec.get('labels')):
        if isinstance(candidate, (list, tuple)):
            labels = []
            for option in candidate:
                if isinstance(option, str):
                    labels.append(option)
                elif isinstance(option, dict):
                    labels.append(str(option.get('name', option.get('label', option.get('id', '')))))
            if labels:
                return labels
    criteria = spec.get('criteria')
    if isinstance(criteria, dict) and 'true' in criteria and 'false' in criteria:
        return list(criteria)
    return None


def true_index(spec, encoded_record, question_name, option_count, default_index=0):
    labels = _option_labels(spec, encoded_record, question_name)
    if labels:
        for i, label in enumerate(labels):
            if label.strip().lower() in ('true', 'yes', 'noul_true'):
                if i >= option_count:
                    raise ValueError(f'{question_name}: true option index {i} exceeds {option_count} logits')
                return i
    if not 0 <= default_index < option_count:
        raise ValueError(f'{question_name}: configured true option index {default_index} exceeds {option_count} logits')
    return default_index


def softmax_probability(logits, index):
    values = [float(x) for x in logits]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('option logits must be finite and non-empty')
    peak = max(values)
    exps = [math.exp(x - peak) for x in values]
    return exps[index] / sum(exps)


def _move_to_device(value, device):
    if hasattr(value, 'to'):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def infer_batch(model, tokenizer, schema, records, device='xpu', default_true_option_index=0, processor=None):
    encode = schema.encode_record
    encoded = []
    for record in records:
        encoded.append(invoke_compatible(
            encode,
            {'record': record, 'example': record, 'item': record, 'data': record,
             'tokenizer': tokenizer, 'tok': tokenizer, 'processor': processor},
            fallback=(record, tokenizer),
        ))
    try:
        import torch
        collate_device = torch.device(device)
    except (ImportError, RuntimeError):
        collate_device = device
    batch = invoke_compatible(
        schema.collate_records,
        {'records': encoded, 'examples': encoded, 'items': encoded, 'batch': encoded,
         'tokenizer': tokenizer, 'tok': tokenizer,
         'pad_token_id': getattr(tokenizer, 'pad_token_id', None), 'device': collate_device},
        fallback=(encoded,),
    )
    batch = _move_to_device(batch, device)
    with _no_grad_context():
        outputs = model(batch)
    logits = outputs[0]
    per_record = normalize_record_logits(logits, records)
    answers = []
    for record, encoded_record, question_logits in zip(records, encoded, per_record):
        row = {}
        for (name, spec), options in zip(record['questions'].items(), question_logits):
            if (spec or {}).get('type') != 'noul':
                raise ValueError(f'{name}: clef_batch_gpu only supports noul questions')
            idx = true_index(spec or {}, encoded_record, name, len(options), default_true_option_index)
            row[name] = softmax_probability(options, idx)
        answers.append(row)
    return answers


def _no_grad_context():
    try:
        import torch
        return torch.inference_mode()
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


def process_requests(requests_path, answers_path, model, tokenizer, schema, *, batch_size=8,
                     limit=0, device='xpu', default_true_option_index=0, processor=None):
    if batch_size <= 0:
        raise ValueError('--batch must be positive')
    records, seen = [], set()
    for row in read_jsonl(requests_path):
        digest = row.get('hash') or row.get('request_hash')
        if not digest:
            digest = request_hash(row['state'], row['questions'])
        if digest in seen:
            continue
        seen.add(digest)
        row['hash'] = digest
        records.append(row)
    answered = load_answered(answers_path)
    pending = [row for row in records if row['hash'] not in answered]
    already_answered = len(records) - len(pending)
    selected = pending
    if limit > 0:
        selected = selected[:limit]
    os.makedirs(os.path.dirname(os.path.abspath(answers_path)), exist_ok=True)
    t0 = time.perf_counter()
    processed = 0
    with open(answers_path, 'a', encoding='utf-8') as dst:
        for offset in range(0, len(selected), batch_size):
            batch = selected[offset:offset + batch_size]
            probabilities = infer_batch(model, tokenizer, schema, batch, device,
                                        default_true_option_index, processor)
            for record, values in zip(batch, probabilities):
                dst.write(json.dumps({'hash': record['hash'], 'answers': values}, separators=(',', ':')) + '\n')
                dst.flush()
                processed += 1
            if device == 'xpu':
                try:
                    import torch
                    torch.xpu.empty_cache()
                except (ImportError, AttributeError, RuntimeError):
                    pass
    elapsed = max(time.perf_counter() - t0, 1e-12)
    return {'seen': len(records), 'already_answered': already_answered,
            'processed': processed, 'seconds': elapsed, 'requests_per_second': processed / elapsed}


def _load_schema(snapshot):
    snapshot = Path(snapshot)
    candidates = [snapshot / 'joint_schema_model.py', Path(__file__).resolve().parent / 'joint_schema_model.py']
    module_file = next((p for p in candidates if p.is_file()), None)
    if module_file is not None:
        sys.path.insert(0, str(module_file.parent))
        spec = importlib.util.spec_from_file_location('joint_schema_model', module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'cannot load model interface module {module_file}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    sys.path.insert(0, str(snapshot))
    try:
        return importlib.import_module('joint_schema_model')
    except ImportError as exc:
        raise RuntimeError('joint_schema_model.py was not found beside this script or in the Clef snapshot') from exc


def _load_model(snapshot, schema):
    import torch
    from transformers import BitsAndBytesConfig

    try:
        import transformers.modeling_utils as modeling_utils
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass
    if not torch.xpu.is_available():
        raise RuntimeError('torch.xpu is unavailable; run this script in the configured Windows A770 environment')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_compute_dtype=torch.bfloat16)
    load_fn = schema.load_release_model
    values = {
        'path': str(snapshot), 'model_path': str(snapshot), 'snapshot': str(snapshot),
        'snapshot_path': str(snapshot), 'release_path': str(snapshot), 'repo_path': str(snapshot),
        'model_id': str(snapshot), 'repo_id': str(snapshot),
        'device': 'xpu', 'map_location': 'xpu',
        'dtype': torch.bfloat16,
        'quantization_config': quant, 'quant_config': quant, 'bnb_config': quant,
        'local_files_only': True, 'trust_remote_code': True,
    }
    # load_release_model sets device_map from device and forwards quantization_config
    # to from_pretrained; it has no vision-encoder skip option in this release.
    extra = {'quantization_config': quant, 'trust_remote_code': True, 'local_files_only': True}
    loaded = invoke_compatible(load_fn, values, fallback=(str(snapshot),), extra_kwargs=extra)
    if isinstance(loaded, dict):
        model = loaded.get('model') or loaded.get('module')
        processor = loaded.get('processor') or loaded.get('tokenizer')
    elif isinstance(loaded, (tuple, list)):
        model = loaded[0]
        processor = loaded[1] if len(loaded) > 1 else None
    else:
        model, processor = loaded, None
    if model is None:
        raise RuntimeError('load_release_model did not return a model')
    if processor is None:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    if hasattr(model, 'eval'):
        model.eval()
    tokenizer = getattr(processor, 'tokenizer', processor)
    return model, tokenizer, processor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--requests', required=True)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--limit', type=int, default=0, help='smoke run: process at most N unanswered unique hashes')
    ap.add_argument('--batch', type=int, default=8, help='records per XPU batch')
    ap.add_argument('--true-option-index', type=int, default=0,
                    help='fallback option index for noul true when the schema carries no option labels')
    a = ap.parse_args(argv)
    if a.limit < 0:
        ap.error('--limit cannot be negative')
    os.environ.setdefault('HF_HOME', r'E:\hf-cache')
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
    schema = _load_schema(snapshot)
    model, tokenizer, processor = _load_model(snapshot, schema)
    result = process_requests(a.requests, a.answers, model, tokenizer, schema,
                              batch_size=a.batch, limit=a.limit, device='xpu',
                              default_true_option_index=a.true_option_index, processor=processor)
    print(f"THROUGHPUT requests={result['processed']} seconds={result['seconds']:.3f} "
          f"requests_per_second={result['requests_per_second']:.3f}", flush=True)
    print(f"processed={result['processed']} already_answered={result['already_answered']} "
          f"unique_requests={result['seen']}", flush=True)


if __name__ == '__main__':
    main()
