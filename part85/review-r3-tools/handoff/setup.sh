set -u
H=$1; B=$2; S=$3
PYV=/home/elisepc/neuron-probe/.venv/bin/python
rm -rf $H/remote.git $H/seed $H/A $H/B
git init -q --bare $H/remote.git && git --git-dir=$H/remote.git symbolic-ref HEAD refs/heads/main
git init -q $H/seed && cd $H/seed && git config user.email t@example.invalid && git config user.name T && echo base > README && git add README && git commit -qm base && git branch -M main && git remote add origin $H/remote.git && git push -q -u origin main
git clone -q $H/remote.git $H/A && git clone -q $H/remote.git $H/B
for c in A B; do git -C $H/$c config user.email t@example.invalid; git -C $H/$c config user.name T; done
# pilot session in clone A
cd $H/A && $PYV -I -B $H/flow.py $B pilot $H > $H/pilot.out 2>$H/pilot.err; echo "pilot flow exit=$?"; tail -2 $H/pilot.err
git --git-dir=$H/remote.git branch -a
