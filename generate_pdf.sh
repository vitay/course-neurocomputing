#!/bin/bash

input_file=$1
output_file="slides/pdf/$(basename -- $input_file .html).pdf"

decktape reveal --size='2048x1536' $input_file $output_file
