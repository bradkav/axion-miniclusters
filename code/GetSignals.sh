ASflag="-AScut"

python simulate_signal.py -profile PL -unperturbed 0 $ASflag
python simulate_signal.py -profile PL -unperturbed 1 $ASflag
python simulate_signal.py -profile NFW -unperturbed 0 $ASflag
python simulate_signal.py -profile NFW -unperturbed 1 $ASflag
