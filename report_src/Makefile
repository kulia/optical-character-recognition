TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
PRE =  $(TEX) -ini -job-name="preamble" "&pdflatex preamble.tex\dump"
BIB = bibtex

 

.PHONY: all view

all: main.pdf

force: 
	touch main.tex
	make

view:
	open main.pdf

main.pdf: main.tex
	$(TEX) main.tex

main.bbl main.blg: main.bib main.aux
	$(BIB) main

main.aux: main.tex
	$(TEX) main.tex

main.bib: main.tex
	$(TEX) main.tex

preamble.fmt: preamble.tex
	$(PRE) preamble.tex

clean:
	rm -f *.acn *.acr *.alg *.aux *.bbl *.bcf *.blg *.dvi *.fdb_latexmk *.fls *.glg *.glo *.gls *.idx *.ilg *.ind *.ist *.lof *.log *.lot *.maf *.mtc *.mtc0 *.nav *.nlo *.out *.pdfsync *.ps *.run.xml *.snm *.synctex.gz *.toc *.vrb *.xdy *.tdo *.brf 
clean-all:
	make clean
	rm -f *.pdf