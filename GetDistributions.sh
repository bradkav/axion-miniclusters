#Set this to determine how many Monte Carlo samples to use
#1000 should be enough to get a good idea of the distributions
NMAXROWS=10000000

#Set ASflag="" to not use the AScut
ASflag="-AScut"
#ASflag=""

echo -----------PL-------------
python3 code/prepare_distributions.py -profile PL --max_rows $NMAXROWS $ASflag
echo -----------PL, circ-------------
python3 code/prepare_distributions.py -profile PL -circ --max_rows $NMAXROWS $ASflag
echo -----------PL, circ, unpert-------------
python3 code/prepare_distributions.py -profile PL -unperturbed 1 -circ --max_rows $NMAXROWS $ASflag

echo -----------NFW-------------
python3 code/prepare_distributions.py -profile NFW --max_rows $NMAXROWS $ASflag
echo -----------NFW, circ-------------
python3 code/prepare_distributions.py -profile NFW -circ --max_rows $NMAXROWS $ASflag
echo -----------NFW, circ, unpert-------------
python3 code/prepare_distributions.py -profile NFW -unperturbed 1 -circ --max_rows $NMAXROWS $ASflag
