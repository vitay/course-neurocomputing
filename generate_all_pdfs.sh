#!/bin/bash

for input_file in `ls docs/slides/*.html`
do
    output_file="slides/pdf/$(basename -- $input_file .html).pdf"
    echo $input_file ">" $output_file
    decktape automatic --size='2048x1536' $input_file $output_file
done