import json, sys
d = json.load(open(sys.argv[1]))
a = d.get("a_attempt2_custom_template", {}); b = d.get("b_main_instruct", {})
print("a: prompts_rendered_with_custom", a.get("prompts_rendered_with_custom_template"), "of", len(a.get("prompt_texts_seen", [])), "| rows_record_custom_sha", a.get("rows_record_custom_sha"), "| first prompt:", repr(a.get("prompt_texts_seen", [""])[0]))
pushes = b.get("pushes", [])
print("b: chat_template_calls", b.get("chat_template_calls"), "| clamp is layers[10].down_proj", b.get("clamp_module_is_layer10_down_proj"), "| pushes", len(pushes), "| push ckpt args", sorted({p["checkpoint_arg"] for p in pushes}), "| first push answers present", pushes[0]["answer_files_present_at_push"] if pushes else None)
print("c:", d.get("c_template_mismatch_main"))
print("d:", d.get("d_step0_push"))
