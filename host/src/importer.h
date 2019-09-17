/* 2019-09-17 Seung-Hun Chung
 * sh.chung@mail.utoronto.ca
 *
 * importer.h
 *
 * Text file parser for importing weights trained in Keras. 
 */

#pragma once

void weight_parser(const char *filename, vector<vector<float>> &weights);
