/****************************************************************************************
//DataListener V4.1
//Author: Jean-Philippe Roberge Ing, M.Sc.A.
//Date: June 9th 2015
//Affiliations: Laboratoire de commande et de robotique (École de technologie supérieure)
//
//Description:  DataListener.cpp - A example script that listens for any kind of data
//              coming from any sensor(s). This starts by subscribing to the topics
//              "/TactileSensor/DynamicData" and "/TactileSensor/DynamicData". It will
//              display any data it receives.
//
//______________________________________________________________________________________
//Version: 1.0 : April 2nd 2015
//Version: 1.1 : June 9th 2015
//
//Last Modified: April 2nd 2015 - Initial release
//               June 9th 2015  - Modified to include acquisition of multiple sensors at
//                                the same time --> e.g. by adding option -sensor 1:10
//                                on the command line
****************************************************************************************/



#include "ros/ros.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Float64.h"
#include "tactilesensors4/StaticData.h"
#include "tactilesensors4/DynamicData.h"
//#include "tactilesensors4/DynamicAndIMUData.h"

void staticdataCallback(const tactilesensors4::StaticData::ConstPtr &msg)
{
    for(int i=0;i<msg->DataS1.taxels.size();i++)
    {
        ROS_INFO("We have received static data: [%i] and [%i] from the sensors", msg->DataS1.taxels[i],msg->DataS2.taxels[i]);
    }
}

void dynamicdataCallback(const tactilesensors4::DynamicData::ConstPtr &msg)
{
    ROS_INFO("We have received dynamic data: [%f] and [%f] from the sensors", msg->DataS1,msg->DataS2);
}

//void imudataCallback(const tactilesensors4::DynamicAndIMUData::ConstPtr &msg)
//{

//    ROS_INFO("We have received dynamic+imu-->dynamic data from sensor 1: [%f]", msg->DynamicDataS1);
//    ROS_INFO("We have received dynamic+imu-->dynamic data from sensor 2: [%f]", msg->DynamicDataS2);

//    // Displaing Raw Data:
//    for(int i=0;i<msg->IMU1RawData.size();i++)
//    {
//        ROS_INFO("We have received  dynamic+imu-->raw data from sensor 1: [%f]", msg->IMU1RawData[i]);
//    }
//    for(int i=0;i<msg->IMU2RawData.size();i++)
//    {
//        ROS_INFO("We have received  dynamic+imu-->raw data from sensor 2: [%f]", msg->IMU2RawData[i]);
//    }
//    ROS_INFO("We have received  dynamic+imu-->raw data from sensor 2y: [%f]", msg->IMU2RawData[4]);
//    // Displaying Quaternions:
//    for(int i=0;i<msg->IMU1Quaternions.size();i++)
//    {
//        ROS_INFO("We have received  dynamic+imu-->quaternions data from sensor 1: [%f]", msg->IMU1Quaternions[i]);
//    }
//    for(int i=0;i<msg->IMU2Quaternions.size();i++)
//    {
//        ROS_INFO("We have received  dynamic+imu-->quaternions data from sensor 2: [%f]", msg->IMU2Quaternions[i]);
//    }

//    //Displaying Euler Angles:
//    for(int i=0;i<msg->IMU1EulerAngles.size();i++)
//    {
//        ROS_INFO("We have received  dynamic+imu-->euler data from sensor 1: [%f]", msg->IMU1EulerAngles[i]);
//    }
//    for(int i=0;i<msg->IMU2EulerAngles.size();i++)
//    {
//        ROS_INFO("We have received  dynamic+imu-->euler data from sensor 2: [%f]", msg->IMU2EulerAngles[i]);
//    }

//}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/TactileSensor4/StaticData", 1000, staticdataCallback);
    ros::Subscriber sub2 = n.subscribe("/TactileSensor4/DynamicData", 1000, dynamicdataCallback);
//    ros::Subscriber sub3 = n.subscribe("/TactileSensor4/DynamicAndIMUData", 1000, imudataCallback);

    ros::spin();
    return 0;
}
