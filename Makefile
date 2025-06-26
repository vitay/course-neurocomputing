all: website

slides_targets := $(wildcard slides/*.qmd)
slides: $(slides_targets)
	quarto render $? --to revealjs

website: 
	cp /Users/vitay/Articles/bibtex/DeepLearning.bib assets/
	cp /Users/vitay/Articles/bibtex/RecurrentNetworks.bib assets/
	quarto render .


clean:
	rm -rf docs