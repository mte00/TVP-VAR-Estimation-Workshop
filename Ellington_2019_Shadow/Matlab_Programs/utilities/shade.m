function []=shade(start,finish,colorstr);

% function []=shade(start,finish,colorstr);
%
%  start and finish are Nx1 vectors of starting and ending years.
%  The function shades between the start and finish pairs using colorstr

if ~exist('colorstr'); colorstr='y'; end;  % default is yellow
curax=axis;
y=[curax(3) curax(4) curax(4) curax(3)];
hold on;
for i=1:length(start);
  x=[start(i) start(i) finish(i) finish(i)];
  fill(x,y,colorstr);
end;
  
% Now, prevent the shading from covering up the lines in the plot.  
h = findobj(gca,'Type','line');
alpha(.5)

h = findobj(gca,'Type','patch');
set(h,'EdgeColor','none');

% This last one makes the tick marks visible
set(gca, 'Layer', 'top')
