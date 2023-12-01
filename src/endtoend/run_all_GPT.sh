#!/bin/bash

python3 endtoend/Galaxy_NLU.py Attraction $1
python3 endtoend/Galaxy_NLU.py Restaurant $1
python3 endtoend/Galaxy_NLU.py Taxi $1
python3 endtoend/Galaxy_NLU.py Hotel $1
python3 endtoend/Galaxy_NLU.py Train $1

python3 endtoend/Galaxy_NLU.py Attraction+Restaurant $1
python3 endtoend/Galaxy_NLU.py Attraction+Hotel $1
python3 endtoend/Galaxy_NLU.py Attraction+Restaurant+Taxi $1
python3 endtoend/Galaxy_NLU.py Attraction+Taxi+Hotel $1
python3 endtoend/Galaxy_NLU.py Attraction+Train $1
python3 endtoend/Galaxy_NLU.py Hotel+Train $1
python3 endtoend/Galaxy_NLU.py Restaurant+Taxi+Hotel $1
python3 endtoend/Galaxy_NLU.py Restaurant+Train $1
python3 endtoend/Galaxy_NLU.py Restaurant+Hotel $1
