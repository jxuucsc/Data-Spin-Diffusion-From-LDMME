#!/usr/bin/env python3
Joule = 4.359744722207185e-18
meter = 5.2917721090380e-11
mbys = 2.1876912636433e6
Ampere = 6.62361823751013e-3
eV = 27.21138624598853
Tesla = 2.3505175675871e5

cm = 5.2917721090380e-9
micrometer = 5.2917721090380e-5
nm = 5.2917721090380e-2
ps = 2.4188843265857e-5

# Ohm meter = J m/s / A^2
Ohm_meter = Joule * mbys / Ampere / Ampere
Ohm = Ohm_meter / meter
cm2byVs = 1e4 * meter * mbys / eV