#ifndef MATLIB_PML_UTILS_H
#define MATLIB_PML_UTILS_H

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <regex>
#include <fstream>

namespace pml {

  // Check whether the given directory exists.
  inline bool dir_exists(const std::string &dirname) {
    DIR *dir = opendir(dirname.c_str());
    if (dir) {
      closedir(dir);
      return true;
    }
    return false;
  }

  // Check whether the given file exists.
  inline bool file_exists(const std::string &filename) {
    FILE *file_ = fopen(filename.c_str(), "rb");
    if (!file_) {
      return false;
    }
    fclose(file_);
    return true;
  }

  // Check wheter the string ends with the given extension.
  inline bool ends_with(const std::string &str,
                        const std::vector<std::string> &extensions) {
    for(auto &ext : extensions){
      std::regex e("(.*)" + ext);
      if (std::regex_match(str, e)){
        return true;
      }
    }
    return false;
  }

  // Create a valid path name from directory and filename.
  inline std::string make_path(const std::vector<std::string> &parts){
    if(parts.empty()){
      return std::string();
    }
    std::string path = parts[0];
    for(int i=1; i<parts.size(); ++i){
      if(path.back() != '/'){
        path.push_back('/');
      }
      path += parts[i];
    }
    return path;
  }

  // Returns a vector of strings containing file names in the directory dirname.
  // The files must end with one of the extensions in the extensions vector.
  // If 'fullpath' is set, the file paths are absolute, starting from '/'
  // Returned files are guaranteed to be in the alphabetical order.
  inline std::vector <std::string> ls(std::string dirname, bool fullpath,
                                      std::vector <std::string> extensions) {
    std::vector <std::string> files;
    if (dir_exists(dirname)) {
      DIR *dir = opendir(dirname.c_str());
      struct dirent *dp;
      while ((dp = readdir(dir)) != NULL) {
        if (ends_with(dp->d_name, extensions)) {
          if (fullpath) {
            files.push_back(make_path({dirname, dp->d_name}));
          } else {
            files.push_back(dp->d_name);
          }
        }
      }
      closedir(dir);
    }
    // Get the files matching the regex.
    sort(files.begin(), files.end());
    return files;
  }

  // Compresses 'src_file' to 'dst_file'.
  inline bool zip(const std::string &src_file, const std::string &dst_file) {
    std::string cmd = "gzip -c " + src_file + " > " + dst_file;
    return !system(cmd.c_str());
  }

  // Decompresses 'src_file' to 'dst_file'.
  inline bool unzip(const std::string &src_file,
                    const std::string &dst_file) {
    std::string cmd = "gunzip -c " + src_file + " > " + dst_file;
    return !system(cmd.c_str());
  }

  // Removes file src_file from the disk.
  inline bool rm(const std::string &src_file) {
    std::string cmd = "rm " + src_file;
    return !system(cmd.c_str());
  }

  // Creates given directory
  inline bool find_or_create(const std::string &dir_name) {
    if (!dir_exists(dir_name)) {
      std::string cmd = "mkdir " + dir_name;
      return !system(cmd.c_str());
    }
    return true;
  }

}
#endif // MATLIB_PML_UTILS_H