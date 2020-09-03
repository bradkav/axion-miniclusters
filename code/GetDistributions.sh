NMAXROWS=100000

echo -----------PL-------------
python prepare_distributions.py -profile PL --max_rows $NMAXROWS
echo -----------PL, circ-------------
python prepare_distributions.py -profile PL -circ --max_rows $NMAXROWS
echo -----------PL, circ, unpert-------------
python prepare_distributions.py -profile PL -unperturbed 1 -circ --max_rows $NMAXROWS

echo -----------NFW-------------
python prepare_distributions.py -profile NFW --max_rows $NMAXROWS
echo -----------NFW, circ-------------
python prepare_distributions.py -profile NFW -circ --max_rows $NMAXROWS
echo -----------NFW, circ, unpert-------------
python prepare_distributions.py -profile NFW -unperturbed 1 -circ --max_rows $NMAXROWS