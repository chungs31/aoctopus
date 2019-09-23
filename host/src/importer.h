/* 2019-09-17 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * importer.h
 *
 * Text file parser for importing weights trained in Keras. 
 */


#ifndef IMPORTER_H
#define IMPORTER_H

#include <vector>

void weight_parser(const char *filename, std::vector<std::vector<float> > &weights);

#endif

