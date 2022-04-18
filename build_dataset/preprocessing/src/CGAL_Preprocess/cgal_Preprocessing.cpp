#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/jet_smooth_point_set.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/remove_outliers.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/bilateral_smooth_point_set.h>
#include <CGAL/tags.h>

#include <utility> // defines std::pair
#include <fstream>
#include <vector>


// types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;  // jet smoothing
typedef Kernel::Point_3 Point_Ex;


typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point_K;
typedef K::Vector_3 Vector_K;
typedef std::pair<Point_K, Vector_K> Pwn;

typedef CGAL::First_of_pair_property_map<Pwn> Point_map;
typedef CGAL::Second_of_pair_property_map<Pwn> Normal_map;
namespace params = CGAL::parameters;

// Concurrency
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "    Usage: " << argv[0] << " [input.xyz/off/ply/las]" << std::endl;
        std::cout << "    Arg1: Infile---txt/xyz/off/ply/las" << std::endl;
        std::cout << "    Arg2: Outfile---txt/xyz" << std::endl;
        std::cout << "    Arg3: Method choose: 0-Remove Outlier /1-Redistration/2-Jet Smoothing/3-ilateral Smoothing" << std::endl << std::endl;
        std::cout << "    For 0-Remove Outliers" << std::endl << std::endl;
        std::cout << "      Argv4 : 0-Using Erase-Remove idiom/1-Using ratio of outliers " << std::endl;
        std::cout << "      Argv5 : nb_neighbors for 0-/percentage for 1- " << std::endl << std::endl;
        std::cout << "    For 1-Redistration" << std::endl ;
        std::cout << "      No develop" << std::endl ;
        
        std::cout << "    For 2-Jet Smoothing" << std::endl ;
        std::cout << "      Argv4 : nb_neighbors " << std::endl << std::endl;
        std::cout << "    For 3-Bilateral Smoothing" << std::endl ;
        std::cout << "      Argv4 : nb_neighbors " << std::endl;
        std::cout << "      Argv5 : sharpness_angle" << std::endl;
        std::cout << "      Argv6 : iter_number" << std::endl;
        
        return EXIT_FAILURE;
    }
    const char* input_file = argv[1];

    unsigned int preprocess
        = (argc < 4 ? 0 : atoi(argv[3]));

    if (preprocess == 0) // Remove Outlier
    {
        std::vector<Point_Ex> points;

        std::ifstream stream(input_file);
        if (!stream ||
            !CGAL::read_xyz_points(stream, std::back_inserter(points),
                CGAL::parameters::point_map(CGAL::Identity_property_map<Point_Ex>())))
        {
            std::cerr << "Error: cannot read file " << input_file << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Read " << points.size() << " point(s)" << std::endl;

        if (points.empty())
            return EXIT_FAILURE;

        unsigned int whichmethod
            = (argc < 5 ? 0 : atoi(argv[4]));
        std::cout << "Remove Outlier Methods is:" << whichmethod << std::endl;
        if (whichmethod == 0)
        {
            // Removes outliers using erase-remove idiom.
            // The Identity_property_map property map can be omitted here as it is the default value.
            const int nb_neighbors = (argc < 6 ? 24 : atoi(argv[5])); // considers 24 nearest neighbor points
            std::cout << "Remove Outlier Number of neighbor is:" << nb_neighbors << std::endl;
            // Estimate scale of the point set with average spacing
            const double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>
                 (points, nb_neighbors);
            // FIRST OPTION //
            // I don't know the ratio of outliers present in the point set
            points.erase(CGAL::remove_outliers<CGAL::Parallel_if_available_tag>
                (points,
                    nb_neighbors,
                    CGAL::parameters::threshold_percent(100.). // No limit on the number of outliers to remove
                    threshold_distance(2. * average_spacing)),points.end()); // Point with distance above 2*average_spacing are considered outliers

            std::cout << " Outlier Removed" << std::endl;
        }
        else {
            const int nb_neighbors = 24;
            // SECOND OPTION //
            // I know the ratio of outliers present in the point set
            const double removed_percentage = 5.0; // percentage of points to remove
            points.erase(CGAL::remove_outliers<CGAL::Parallel_if_available_tag>
                (points,
                    nb_neighbors,
                    CGAL::parameters::threshold_percent(removed_percentage). // Minimum percentage to remove
                    threshold_distance(0.)), // No distance threshold (can be omitted)
                points.end());
            std::cout << " Outlier Removed" << std::endl;
        }
        // Optional: after erase(), use Scott Meyer's "swap trick" to trim excess capacity
        std::vector<Point_Ex>(points).swap(points);
        
        std::ofstream f(argv[2]);
        f.precision(17);
        CGAL::write_xyz_points(f, points);
        f.close();
    }

    else if (preprocess == 1) // Registration
    {
        std::cout << "I dont know how to registration !!!" << std::endl;
    }

    else if (preprocess == 2) // Jet Smoothing
    {
        std::vector<Point_Ex> points;
        std::ifstream stream(input_file);
        if (!stream ||
            !CGAL::read_xyz_points(stream, std::back_inserter(points),
                CGAL::parameters::point_map(CGAL::Identity_property_map<Point_Ex>())))
        {
            std::cerr << "Error: cannot read file " << input_file << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Read " << points.size() << " point(s)" << std::endl;

        if (points.empty())
            return EXIT_FAILURE;

        unsigned int neighborpoints
            = (argc < 5 ? 24 : atoi(argv[4]));
        std::cout << "neighbor points is:" << neighborpoints << std::endl;
        CGAL::jet_smooth_point_set<CGAL::Sequential_tag>(points, neighborpoints);
        std::cout << " Jet Smoothing Finished !!!." << std::endl;
        
        std::ofstream f(argv[2]);
        f.precision(17);
        CGAL::write_xyz_points(f, points);
    }

    else if (preprocess == 3) // bilateral Smoothing  Need to input with normals 
    {
        // Reads a .xyz point set file in points[] * with normals *.
        std::vector<Pwn> points2;
        std::ifstream stream(input_file);
        if (!stream ||
            !CGAL::read_xyz_points(stream,
                std::back_inserter(points2),
                CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
                normal_map(CGAL::Second_of_pair_property_map<Pwn>())))
        {
            std::cerr << "Error: cannot read file " << input_file << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Read " << points2.size() << " point(s)" << std::endl;

        // Algorithm parameters
           // size of neighborhood. The bigger the smoother the result will be.
           // This value should bigger than 1.
        unsigned int neighborpoints
            = (argc < 5 ? 120 : atoi(argv[4]));
        
        double sharpness_angle  // control sharpness of the result.// The bigger the smoother the result will be
                 = (argc < 6 ? 120 : atof(argv[5]));
                                     
        int iter_number         // number of times the projection is applied
                = (argc < 7 ? 3: atoi(argv[6]));
        std::cout << "neighbor points is:" << neighborpoints << "    sharpness anlge is:" << sharpness_angle << "    iter number is:" << iter_number << std::endl;
        for (int i = 0; i < iter_number; ++i)
        {
            /* double error = */
            CGAL::bilateral_smooth_point_set <Concurrency_tag>(
                points2,
                neighborpoints,
                CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
                normal_map(CGAL::Second_of_pair_property_map<Pwn>()).
                sharpness_angle(sharpness_angle));
        }
        std::ofstream out(argv[2]);
        out.precision(17);
        if (!out ||
            !CGAL::write_xyz_points(
                out, points2,
                CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()).
                normal_map(CGAL::Second_of_pair_property_map<Pwn>())))
        {
            return EXIT_FAILURE;
        }
    }

    else // Handle error
    {
        std::cerr << "Error: invalid reconstruction id: " << preprocess << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}