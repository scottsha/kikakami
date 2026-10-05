#include <iostream>
#include <vector>
#include <filesystem>

#include <boost/program_options.hpp>

#include "io_utilities.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {

    std::filesystem::path path_in;
    std::filesystem::path path_out;
    bool is_key4_tagging = false;

    // ---------------------------------------
    // COMMAND LINE OPTION SETUP
    // ---------------------------------------
    po::options_description io_options("Translate Geometry Formats");
    io_options.add_options()
            ("help,h", "Produce help message")
            ("input,i", po::value<std::filesystem::path>(&path_in)->required(), "Input file path")
            ("output,o", po::value<std::filesystem::path>(&path_out)->required(), "Output file path")
            ;

    po::variables_map vm;
    try {
        auto parsed = po::command_line_parser(argc, argv).options(io_options).run();
        po::store(po::parse_command_line(argc, argv, io_options), vm);
        po::notify(vm);
    } catch (boost::program_options::error &err) {
        if (vm.count("help") || vm.count("h")) {
            std::cout << io_options << "\n";
        } else {
            std::cerr << err.what() << std::endl;
        }
        return 1;
    }

    if (vm.count("help") || vm.count("h")) {
        std::cout << io_options << "\n";
        return 1;
    }

    // ---------------------------------------
    // IO printouts
    // ---------------------------------------
    std::cout << "Input at: " << path_in << std::endl;
    std::cout << "Output at: " << path_out << std::endl;

    return 0;
}


