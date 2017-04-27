function [fig, figObj] = newFig(fig)
	fig = fig + 1;
	figObj = figure(fig);
	clf(fig);
	set(0,'defaultaxesfontname','times');
	box on 