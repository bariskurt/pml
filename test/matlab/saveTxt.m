function saveTxt (filename, X)

  fid = fopen(filename,'w');
  fprintf(fid, '%d\n', ndims(X));
  for i = 1:ndims(X),
    fprintf(fid, '%d\n', size(X, i));
  end
  fclose(fid);
  dlmwrite(filename, X(:), '-append', 'precision', 6);
end
