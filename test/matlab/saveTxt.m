## Copyright (C) 2016 Baris Kurt
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} saveTxt (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Baris Kurt <bariskurt@GLaDOS>
## Created: 2016-10-10

function saveTxt (filename, X)

  fid = fopen(filename,'w');
  fprintf(fid, '%d\n', ndims(X))
  for i = 1:ndims(X),
    fprintf(fid, '%d\n', size(X, i))
  end
  fclose(fid)
  dlmwrite(filename, X(:), '-append', 'precision', 6)
end
