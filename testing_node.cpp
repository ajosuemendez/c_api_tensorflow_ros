#include <ros/ros.h>
#include <stdio.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <numpy_tutorial/Frequency.h>
#include <tensorflow/c/c_api.h>
//#include <numpy_tutorial/tf_utils.h>
#include <iostream>
#include <iostream>
#include <vector>
//#include <time.h>
#include <typeinfo>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <rospy_tutorials/Floats.h>
#include <chrono>

class DisplayInfo {
  private:
  ros::Publisher pub;
  ros::Subscriber sub;

  private:
    TF_Buffer* m_buffer;
    TF_Session* m_sess;
    TF_SessionOptions* m_sessOptions;
    TF_Graph* m_graph;
    TF_Status* m_status;

    TF_Tensor* m_input_tensor;
    TF_Tensor* m_output_tensor = nullptr;

    TF_Output m_output_op;
    TF_Output m_input_op;

    std::vector<float> output_vals;
    std::map<const int, const char*> classNames;

    numpy_tutorial::Frequency frequency;

    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    std::chrono::time_point<std::chrono::high_resolution_clock> t2;

    std::vector<std::chrono::duration<double>> predictionTimeArr;

    bool isFirstLoop = true;

    bool isEfficientNetModel = false;

    //const char* inputOpTensorName = "serving_default_input_1"; //input tensor name for EfficientNetModel
    const char* inputOpTensorName = "serving_default_conv2d_input"; //input tensor name for FoodModel

    const char* outputOpTensorName = "StatefulPartitionedCall";

    // isEfficientNetModel, inputOpTensorName, loadGraph


  public:
  DisplayInfo(ros::NodeHandle *nh) {
    this->setClasses();
    this->t1 = std::chrono::high_resolution_clock::now();
    this->loadGraph("/home/mendez/PycharmProjects/pythonProject1/saved_model/model_1");
    //this->loadGraph("/home/mendez/EfficientNetB7_model");
    this->getInOutOperations();
    this->frequency.count = 0;
    pub = nh->advertise<numpy_tutorial::Frequency>("/shows_message", 10);
    sub = nh->subscribe("/send_image_data", 1000, &DisplayInfo::callback_trigger, this);
  }


  void setClasses(){
    this->classNames.insert(std::pair<const int, const char*>(0, "Chicken Curry"));
    this->classNames.insert(std::pair<const int, const char*>(1, "Chicken Wings"));
    this->classNames.insert(std::pair<const int, const char*>(2, "Fried Rice"));
    this->classNames.insert(std::pair<const int, const char*>(3, "Grilled Salmon"));
    this->classNames.insert(std::pair<const int, const char*>(4, "Hamburger"));
    this->classNames.insert(std::pair<const int, const char*>(5, "Ice Cream"));
    this->classNames.insert(std::pair<const int, const char*>(6, "Pizza"));
    this->classNames.insert(std::pair<const int, const char*>(7, "Ramen"));
    this->classNames.insert(std::pair<const int, const char*>(8, "Steak"));
    this->classNames.insert(std::pair<const int, const char*>(9, "Sushi"));
  }
  

  void callback_trigger(const rospy_tutorials::Floats& msg) {
    if (this->isFirstLoop) {
      this->t2 = std::chrono::high_resolution_clock::now();
      this->isFirstLoop = false;
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();

    this->createInputTensor(msg);
    this->runSession();
    this->frequency.count++;
    pub.publish(this->frequency);
    auto t4 = std::chrono::high_resolution_clock::now();

    /*
    std::cout << "DIF T2-T1: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    std::cout << "DIF T3-T2: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count()
    << " milliseconds\n";
*/
    std::chrono::duration<double> elapsed_1 = this->t2 - this->t1;
    std::chrono::duration<double> elapsed_2 = t4 - t3;

    this->predictionTimeArr.push_back(elapsed_2);


    std::cout << " PREDICTION TIME ARR: [";
    for(auto & element : predictionTimeArr) {
      std::cout << element.count() << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << " LOAD MODEL TIME: " << elapsed_1.count() << std::endl;
    std::cout << " PREDICTION TIME: " << elapsed_2.count() << std::endl;
    std::cout << "---------------------------------------" << std::endl;


  }

  void loadGraph(const char* graph_path)
    {
      this->m_graph = TF_NewGraph();
      this->m_status = TF_NewStatus();
      this->m_buffer = NULL;
      const char* tags = "serve";
      this->m_sessOptions = TF_NewSessionOptions();

      int ntags = 1;
      
      this->m_sess = TF_LoadSessionFromSavedModel(this->m_sessOptions, this->m_buffer, graph_path, 
                                                  &tags, ntags, this->m_graph, NULL, this->m_status);

      if (TF_GetCode(this->m_status) == TF_OK)
        {
          printf("Tensorflow 2x Model loaded OK\n");
        }
        else
        {
          printf("%s", TF_Message(this->m_status));
        }
      
      TF_DeleteSessionOptions(this->m_sessOptions);
      TF_DeleteStatus(this->m_status);  
    }

  void getInOutOperations(){
    this->m_output_op = { TF_GraphOperationByName(this->m_graph, this->outputOpTensorName), 0};
    if (this->m_output_op.oper == NULL) {
      printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall_0\n");
    }
    else{
       printf("TF_GraphOperationByName StatefulPartitionedCall_0 is OK\n");
      }

    this->m_input_op = { TF_GraphOperationByName(this->m_graph, this->inputOpTensorName), 0};
    if (this->m_input_op.oper == NULL) {
      printf("ERROR: Failed TF_GraphOperationByName serving_default_conv2d_input_0\n");
    }
    else{
       printf("TF_GraphOperationByName serving_default_conv2d_input_0 is OK\n");
      }
  }

  void createInputTensor(const rospy_tutorials::Floats& msg){

    const int channels = 3;
    const int rows = 224;
    const int columns = 224;
    const std::vector<std::int64_t> input_dims = {1, columns, rows, channels};

    std::size_t input_len = (columns * rows * channels) * sizeof(float);
    
    this->m_input_tensor = TF_AllocateTensor(TF_FLOAT, input_dims.data(), input_dims.size(), input_len);
    input_len = std::min(input_len, TF_TensorByteSize(this->m_input_tensor));

    //Getting Image Data
    std::vector<float> imageData = msg.data;

    if (input_len != 0)
      {
          memcpy(TF_TensorData(this->m_input_tensor), imageData.data(), input_len);
          std::cout <<  "input tensor created" << std::endl;
      }
    else
      {
          std::cout << "input tensor error " << std::endl;
          TF_DeleteTensor(this->m_input_tensor);
      }
  }

  
  void runSession(){
    this->m_status = TF_NewStatus();

    TF_SessionRun(this->m_sess, nullptr, &this->m_input_op, &this->m_input_tensor, 1,
                  &this->m_output_op, &this->m_output_tensor, 1, 
                  nullptr, 0, nullptr, this->m_status);

    if (TF_GetCode(this->m_status) != TF_OK)
          {
            std::cout << "Error run session code="<<TF_GetCode(this->m_status);
          }

    //Extracting the data
    auto data = static_cast<float*>(TF_TensorData(this->m_output_tensor));
  

    //Translating the data into human readable data
    const char* predictedClassName = this->getClassName(data);

    if (predictedClassName == nullptr){
      std::cout << "Something went wrong " << std::endl;
      return;
    }
   // std::cout << "Class: " << predictedClassName <<std::endl;
    this->frequency.className = predictedClassName;
    
  }


  const char* getClassName(float* data){
    int max_index;
    float max_value = 0.0;
    float total_sum = 0;
    int size = classNames.size();

    if (this->isEfficientNetModel) {
      size = 1000;
    }
    
    for (int x = 0; x < size; x++){
      
      if (data[x] > max_value){
        max_index = x;
        max_value = data[x];
      }
      total_sum += data[x];
    }
    
    

    //CHECKSUM
    if (total_sum < 0.99) {
      return nullptr;
    }
    if (this->isEfficientNetModel) {

      return this->classNames[6];
    }
    return this->classNames[max_index];
  }
/*
  void closeSession(void)
  {
      TF_CloseSession(this->m_sess, this->m_status);
      if (TF_GetCode(this->m_status) != TF_OK)
      {
          std::cout << "Error close session";
      }
      else{
        std::cout << "close session Successful";
      }
  }


    void deleteSession(void)
    {
        TF_DeleteSession(this->m_sess, this->m_status);
        if (TF_GetCode(this->m_status) != TF_OK)
        {
            std::cout << "Error delete session";
        }
        TF_DeleteSessionOptions(this->m_sessOptions);
        TF_DeleteStatus(this->m_status);
        TF_DeleteGraph(this->m_graph);
    }
*/
};

int main (int argc, char **argv)
{
    ros::init(argc, argv, "number_counter");
    ros::NodeHandle nh;
    DisplayInfo nc = DisplayInfo(&nh);
    ros::spin();
}
