#!/bin/sh
for f in *.tex; do
    ff=$(echo $f | sed -e 's/\(.tex\)*$//g')
    pdflatex -jobname=$ff "\def\inputfile{$ff.tex}\input{../generate_figures.tex}"
done