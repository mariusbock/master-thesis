#include <cstddef>
#include <string>
#include <fstream>
#include <andres/graph/hdf5/hdf5.hxx>
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"
#include <tclap/CmdLine.h>
#include "Timer.hpp"
#include "utils.hxx"
//#include "andres/ilp/cplex.hxx"


using namespace andres::graph;

typedef double value_type;
typedef std::size_t size_type;

struct Parameters {
  std::vector<value_type> theta_ { -0.5, 2.0 };
  std::string inputFileName_;
  std::string outFileName_;  
  value_type bias_ { .5 };
};

inline void
parseCommandLine(
    int argc, 
    char** argv, 
    Parameters& parameters
) {
    try {
        TCLAP::CmdLine tclap("mc higher order", ' ', "1.0");
        TCLAP::ValueArg<std::string> argInputFileName("i", "input", ".txt", true, parameters.inputFileName_, "INPUT", tclap);
        TCLAP::ValueArg<std::string> argOutFileName("o", "hdf5-file", "hdf5 file (output)", true, parameters.outFileName_, "OUTPUT_HDF5_FILE", tclap);
        TCLAP::MultiArg<value_type> argTheta("t", "theta", "Coefficients of a polynomial (starting with the constant term). Use multiple, e.g. -t -0.5 -t 2.0", false, "THETA-i", tclap);
        TCLAP::ValueArg<value_type> argBias("b", "bias", "logistic prior probability for pixels to be cut", false, parameters.bias_, "BIAS", tclap);
        
        tclap.parse(argc, argv);

        parameters.inputFileName_ = argInputFileName.getValue();
        parameters.outFileName_ = argOutFileName.getValue();
        
        if(argTheta.isSet())
            parameters.theta_ = argTheta.getValue();
        if(parameters.theta_.size() == 0)
            std::cerr<<"internal error."<<std::endl;
        parameters.bias_ = argBias.getValue();
    }
    catch(TCLAP::ArgException& e) {
      std::cerr<<e.error()<<std::endl;
    }
}

void loadProblem(const Parameters& parameters, andres::graph::Graph<>& graph,std::vector<double>& weights){
  
  std::string aFilename = parameters.inputFileName_;
  std::ifstream aIn(aFilename.c_str());
  int noNodes;
  aIn>>noNodes;
  int aSize;
  aIn >> aSize;
 
  graph.assign(noNodes);
  std::cout<<aSize<<" "<<noNodes<<std::endl;
  for (int i = 0; i < aSize; ++i) {
    std::size_t dummy;
    aIn >> dummy;
    std::size_t v1 = dummy;
    aIn >> dummy;
    std::size_t v2 = dummy;
    float fdummy;
    aIn >> fdummy;
    float dist = fdummy;

    float weight;
    float bp = parameters.bias_;
    //float b = ::log(std::max(0.001f,bp)/std::max(0.001f,(1-bp)));
    float b = ::log(std::max(0.00001f,bp)/std::max(0.00001f,(1-bp)));

    std::pair<bool,std::size_t> edge_test;
    edge_test = graph.findEdge(v1,v2);
    //weight =::log(std::max(0.001f,(dist))/std::max(0.001f,(1-(dist))))+b;
    weight =::log(std::max(0.00001f,(dist))/std::max(0.00001f,(1-(dist))))+b;
    if(edge_test.first) {
        // uncomment if block if min or max setup wanted (see thesis for more details)
        if((weight>0.5) && (weights[edge_test.second]>0.5)) {
            weights[edge_test.second] = std::max(weights[edge_test.second], static_cast<double>(weight));
        } else if((weight<0.5) && (weights[edge_test.second]<0.5)) {
            weights[edge_test.second] = std::min(weights[edge_test.second], static_cast<double>(weight));
        } else {
            weights[edge_test.second] = 0.0;
        }
        // change to max if max setup wanted, or change to min if min setup wanted
        //weights[edge_test.second] = std::max(weights[edge_test.second], static_cast<double>(weight));
    } else {
        graph.insertEdge(v1,v2);
        weights.push_back(static_cast<double>(weight));
    }
  }
}
 


void solveProblem(const Parameters& parameters,andres::graph::Graph<>& graph,std::vector<double>& weights){
  //std::vector<double> weights(graph.numberOfEdges(),0.00);
  std::cout<<graph.numberOfEdges()<<" "<<std::endl;
  
  std::vector<char> edge_labels(graph.numberOfEdges(), 0);
  Timer t;
  t.start();
  andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);
  andres::graph::multicut::kernighanLin(graph, weights, edge_labels, edge_labels);
  t.stop();
  double energy = 0;
  for(size_type v = 0; v < graph.numberOfEdges(); ++v) {
    energy +=(edge_labels[v])*weights[v];
    //std::cout<<static_cast<float>(edge_labels[v])<<" "<<std::flush;
  }
  std::cout<<"energy "<<energy<<std::endl;
  std::vector<size_t> vertex_labels(graph.numberOfVertices());
  
  edgeToVertexLabels(graph, edge_labels, vertex_labels);
  std::cout << "saving multicut problem to file: " << parameters.outFileName_ << std::endl;
  //for(size_type v = 0; v < graph.numberOfVertices(); ++v) {
  //  std::cout<<static_cast<float>(vertex_labels[v])<<" "<<std::flush;
  //}
  //auto file = andres::graph::hdf5::createFile(parameters.outFileName_);
  //andres::graph::hdf5::save(file, "labels", { vertex_labels.size() }, vertex_labels.data());
  //andres::graph::hdf5::save(file, "energy-value", energy);
  //andres::graph::hdf5::save(file, "running-time", t.get_elapsed_seconds());
  //andres::graph::hdf5::closeFile(file);
  std::ofstream new_file;
  new_file.open(parameters.outFileName_);
  for(int i = 0; i<vertex_labels.size(); i++)
    new_file << vertex_labels[i] << "\n";
  new_file.close();
}


int main(int argc, char** argv) {
  Parameters parameters;
  parseCommandLine(argc, argv, parameters);
  andres::graph::Graph<> graph;
  std::vector<double> weights;
  loadProblem(parameters, graph, weights);
  solveProblem(parameters,graph,weights);
  return 0;
}
