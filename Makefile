all: notes slides

slides_targets := $(wildcard slides/*.qmd)
notebook_targets := $(wildcard slides/exercises/*.ipynb)
slides: $(slides_targets) $(notebook_targets)
	quarto render $? --to revealjs


notes_targets := $(wildcard notes/notes/*.qmd)
notes: $(notes_targets)
	quarto render $?  --to html


clean:
	rm -rf docs
	mkdir docs
	mkdir docs/slides
	quarto render slides --to revealjs
	quarto render notes --to html

export: 
	git add *
	git commit -a -m "`date`"
	git push origin master
	
