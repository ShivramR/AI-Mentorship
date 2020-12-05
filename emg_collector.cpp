#define _USE_MATH_DEFINES
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <time.h>
#include <chrono>
#include <ctime>
#include "myo.hpp"

auto name = "rest";

class DataCollector : public myo::DeviceListener {
public:
    DataCollector()
    {
        openFiles();
    }

    void openFiles() {
        auto timestamp = name;
        
        if (emgFile.is_open()) {
            emgFile.close();
        }
        std::ostringstream emgFileString;
        emgFileString << "emg-" << timestamp << ".csv";
        emgFile.open(emgFileString.str(), std::ios::out);
        emgFile << "timestamp,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8" << std::endl;

    }

    void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg)
    {
        
        emgFile << timestamp
        << ',' << static_cast<int>(emg[0])
        << ',' << static_cast<int>(emg[1])
        << ',' << static_cast<int>(emg[2])
        << ',' << static_cast<int>(emg[3])
        << ',' << static_cast<int>(emg[4])
        << ',' << static_cast<int>(emg[5])
        << ',' << static_cast<int>(emg[6])
        << ',' << static_cast<int>(emg[7]) << std::endl;
        

    }

  

    void onConnect(myo::Myo *myo, uint64_t timestamp, myo::FirmwareVersion firmwareVersion) {
        //Reneable streaming
        myo->setStreamEmg(myo::Myo::streamEmgEnabled);
        openFiles();
    }

    // Helper to print out accelerometer and gyroscope vectors
    void printVector(std::ofstream &file, uint64_t timestamp, const myo::Vector3< float > &vector) {
        file << timestamp
            << ',' << vector.x()
            << ',' << vector.y()
            << ',' << vector.z()
            << std::endl;
    }

    // The files we are logging to
    std::ofstream emgFile;

};

int main(int argc, char** argv)
{
    // We catch any exceptions that might occur below -- see the catch statement for more details.
    try {

    // First, we create a Hub with our application identifier. Be sure not to use the com.example namespace when
    // publishing your application. The Hub provides access to one or more Myos.
    myo::Hub hub("com.undercoveryeti.myo-data-capture");

    std::cout << "Attempting to find a Myo..." << std::endl;

    myo::Myo* myo = hub.waitForMyo(10000);

    // If waitForMyo() returned a null pointer, we failed to find a Myo, so exit with an error message.
    if (!myo) {
        throw std::runtime_error("Unable to find a Myo!");
    }

    // We've found a Myo.
    std::cout << "Connected to a Myo armband! Logging to the file system. Check the folder this appliation lives in." << std::endl << std::endl;

    // Next we enable EMG streaming on the found Myo.
    myo->setStreamEmg(myo::Myo::streamEmgEnabled);

    // Next we construct an instance of our DeviceListener, so that we can register it with the Hub.
    DataCollector collector;

    // Hub::addListener() takes the address of any object whose class inherits from DeviceListener, and will cause
    // Hub::run() to send events to all registered device listeners.
    hub.addListener(&collector);
    // Finally we enter our main loop.
    while (1) {
        // In each iteration of our main loop, we run the Myo event loop for a set number of milliseconds.
        // In this case, we wish to update our display 50 times a second, so we run for 1000/20 milliseconds.
        hub.run(260);
//        myo->vibrate(myo::Myo::vibrationShort);
    }

    // If a standard exception occurred, we print out its message and exit.
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Press enter to continue.";
        std::cin.ignore();
        return 1;
    }
}
