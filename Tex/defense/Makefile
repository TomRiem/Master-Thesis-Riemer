TMPFILES	= *.{log,aux,toc,out,lof,lot,snm,nav,vrb,pdf,bak}
LATEX		= pdflatex


default:	beamer

all:		beamer handout notes dualmon


beamer:		beamer.pdf
handout:	handout.pdf
notes:		notes.pdf
dualmon:	dualmon.pdf
article:	article.pdf


%.pdf: %.tex main.tex $(wildcard inhalt/*) $(wildcard bilder/*)
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<

clean:
	rm -f $(TMPFILES)
