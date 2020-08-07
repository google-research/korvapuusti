#!/bin/bash

grep "eval: " $1 | cut -d ":" -f2-| sed 's/\[//g' | sed 's/\]//g'| sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' &> $2
