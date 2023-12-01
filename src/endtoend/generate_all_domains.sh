#!/bin/bash
bash endtoend/generate_multiwoz_select.sh Attraction $1 $2 $3 $4 $5 $6 $7 $8

bash endtoend/generate_multiwoz_select.sh Restaurant $1 $2 $3 $4 $5 $6 $7 $8

bash endtoend/generate_multiwoz_select.sh Taxi $1 $2 $3 $4 $5 $6 $7 $8

bash endtoend/generate_multiwoz_select.sh Hotel $1 $2 $3 $4 $5 $6 $7 $8

bash endtoend/generate_multiwoz_select.sh Train $1 $2 $3 $4 $5 $6 $7 $8

bash endtoend/generate_multiwoz_select.sh Attraction+Restaurant $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Attraction+Restaurant+Taxi $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Attraction+Hotel $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Attraction+Taxi+Hotel $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Attraction+Train $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Restaurant+Hotel $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Restaurant+Train $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Restaurant+Taxi+Hotel $1 $2 $3 $4 $5 $6 $7 $8
bash endtoend/generate_multiwoz_select.sh Hotel+Train $1 $2 $3 $4 $5 $6 $7 $8

python3 endtoend/merger.py $1 $8 $9
