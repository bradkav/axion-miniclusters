#Set ASflag="" to ignore the AS cut
ASflag="-AScut"

#Number of signal events to generate
Nsig=1e5

python code/simulate_signal.py -profile PL -unperturbed 0 $ASflag -Ne $Nsig
python code/simulate_signal.py -profile PL -unperturbed 1 $ASflag -Ne $Nsig
python code/simulate_signal.py -profile NFW -unperturbed 0 $ASflag -Ne $Nsig
python code/simulate_signal.py -profile NFW -unperturbed 1 $ASflag -Ne $Nsig
