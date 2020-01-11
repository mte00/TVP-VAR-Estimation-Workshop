function []=shadenber(blah);

% shadenber.m
%
%  Routine to shade the nberdates in a figure

load nberdates;  % start and finish contain the recession periods, back to 1857
curax=axis;
indx1=find(finish>curax(1));  % First recession to include;
indx2=find(start<curax(2));  % Last recession to include;
indx1=indx1(1);
indx2=indx2(length(indx2));
if start(indx1)<curax(1);
  start(indx1)=curax(1);
end;
if finish(indx2)>curax(2);
  finish(indx2)=curax(2);
end;

colorstr=[159 182 205]/256;

shade(start(indx1:indx2),finish(indx1:indx2),colorstr);
