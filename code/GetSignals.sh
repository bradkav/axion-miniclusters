ASflag="-AScut"

Nsig=1e4

python simulate_signal.py -profile PL -unperturbed 0 $ASflag -Ne $Nsig
python simulate_signal.py -profile PL -unperturbed 1 $ASflag -Ne $Nsig
python simulate_signal.py -profile NFW -unperturbed 0 $ASflag -Ne $Nsig
python simulate_signal.py -profile NFW -unperturbed 1 $ASflag -Ne $Nsig
