#include "Analysis/ana_processor.h"

#include "supera_light.h"

std::string base_name(std::string const & path)
{
  return path.substr(path.find_last_of("/\\") + 1);
}

std::string remove_extension(std::string const & filename)
{
  std::string::size_type const p(filename.find_last_of('.'));
  return p > 0 && p != std::string::npos ? filename.substr(0, p) : filename;
}


int main(int argc, char const *argv[]) {

    if (argc == 1){
        std::cout << "Must include an output file!" << std::endl;
        return 1;
    }

    std::string output_file_target_name(argv[1]);
    std::string::size_type const p(output_file_target_name.find_last_of('.'));
    if (p > 0){
        output_file_target_name = output_file_target_name.substr(0,p);
    }
    // std::string basename = output_file_target_name.substr(output_file_target_name.find_last_of("//") + 1);

    // remove_extension(basename(output_file_target_name));

    output_file_target_name += std::string("_larcv.h5");

    std::cout << output_file_target_name << std::endl;

    galleryfmwk::ana_processor ap;

    for (size_t i = 1; i < argc; i ++){
        std::string file = std::string(argv[i]);
        ap.add_input_file(file);
    }

    // std::cout << "Beginning processing of file " << file << std::endl;

    // Create the ana processor:


        // base_name = os.path.basename(f)
        //
        // out_dir = './'
        // out_name = out_dir + base_name.replace(".root", "_larcv.h5")
        // # supera_light = supera.supera_light(detClocks.DataForJob())

    auto supera_light = supera::supera_light();
    supera_light.set_output_file(output_file_target_name);

        // # Attach an analysis unit ... here we use a base class which do

    ap.add_process(&supera_light);
    // print(f"Beging Processing run.")

    ap.run();

    return 0;
}
