package org.deeplearning4j.examples.test;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Administrator on 5/2/2017.
 * Based on 'BasicCSVClassifier'
 */
public class artificial_lighting_mixed {
    private static Logger log = LoggerFactory.getLogger(test.class);

//    private static Map<Integer,String> time = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/time.csv");
//    private static Map<Integer,String> outdoor_condition = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/outdoor_condition.csv");
//    private static Map<Integer,String> length_of_leave = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/length_of_leave.csv");
//    private static Map<Integer,String> occupancy = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/occupancy.csv");
//    private static Map<Integer,String> previous_light_state = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/previous_light_state.csv");
//    private static Map<Integer,String> light_state = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/light_state.csv");

    //----------------------------------------------------------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        System.out.println("main started");
//        boolean gpu_used = Nd4j.create(1).getClass().equals(org.nd4j.linalg.jcublas.JCublasNDArray.class);
//        if (gpu_used) System.out.println("GPU use detected.");
//        else System.out.println("WARNING: no GPU use detected, base class being used is " + Nd4j.create(1).getClass());


        try {
            DataNormalization normalizer;
            MultiLayerNetwork model;
            //--------------------------------
            String nn_dir = "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting";
            String model_name = "artificial_lighting_hunt_lum";
            String nn_file_name = nn_dir + "/" + model_name + "_net.dat";
            String nn_file_name_model = nn_dir + "/" + model_name + "_net_model.dat";

            System.out.println("Loading archived data from file " + nn_file_name + " and " + nn_file_name_model + " ...");
            FileInputStream fileIn = new FileInputStream(nn_file_name); // load archived 'data'
            ObjectInputStream in = new ObjectInputStream(fileIn);
            normalizer = (DataNormalization) in.readObject();
            in.close();
            fileIn.close();
            model = ModelSerializer.restoreMultiLayerNetwork(nn_file_name_model);

            //--------------------------------

            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            int num_features=4;
            int labelIndex = num_features;     //3 values in each row of the synthetic_hunt_data.csv: 1 input features followed by an integer label (class) index. Labels are the 3rd value (index 2) in each row
            int numClasses = 2;     //2 classes (light off/light on). Classes have integer values 0 or 1
            int epochs = 100; //33; // 100
            int iterations = 1;
            INDArray tmp;

            int batchSizeTraining = 100000;    //artificial lighting Hunt data set: 100,000 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
            DataSet syntheticVRData = readCSVDataset(
                "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum.csv",
                batchSizeTraining, labelIndex, numClasses);

            System.out.println("remove feature 3(OutdoorCondition)..."); // Occupancy, Leaving, WorkingAreilluminance(HUNT)
            tmp = Nd4j.concat(1, syntheticVRData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(0,2)), syntheticVRData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(3,4)));
            syntheticVRData.setFeatures(tmp.dup());
            num_features = num_features - 1;//*/

            /*DataSet actualData = readCSVDataset(
                "/DataExamples/artificial_lighting/ActualData_for_Prediction_no_hidden.csv",
                batchSizeTraining, labelIndex, numClasses);//*/

            SplitTestAndTrain split_syntheticVR = syntheticVRData.splitTestAndTrain(.90); // 9000/10,000 train
            DataSet trainingData_syntheticVR = split_syntheticVR.getTrain();
            DataSet testData_syntheticVR = split_syntheticVR.getTest(); // this is the data we want to classify
            Nd4j.writeNumpy(testData_syntheticVR.getFeatureMatrix(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum_in.csv", ",");
            Nd4j.writeNumpy(testData_syntheticVR.getLabels(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum_labels.csv", ",");

            normalizer.transform(syntheticVRData);     //Apply normalization to the training data
            //--------------------------------
            /*System.out.println("Hunt data(synthetic_hunt_data_continuous):");
            System.out.println("Read Hunt dataset...");

            labelIndex = 1;
            DataSet huntData = readCSVDataset(
                "/DataExamples/artificial_lighting/synthetic_hunt_data_continuous.csv",
                batchSizeTraining, labelIndex, numClasses);

            System.out.println("Add " + (num_features-1) + " blank features before real feature...");
            tmp = Nd4j.concat(1,Nd4j.zeros(huntData.getFeatures().rows(), (num_features-1)),huntData.getFeatures());
            huntData.setFeatures(tmp.dup());

            System.out.println("Split testing and training sets...");
            SplitTestAndTrain split = huntData.splitTestAndTrain(.05); // 500/100,000 train
            DataSet trainingData = split.getTrain();
            DataSet testData = split.getTest(); // this is the data we want to classify
            Nd4j.writeNumpy(testData.getFeatureMatrix(),"C:/Users/Administrator/Documents/GitHub/dl4j-examples/dl4j-cuda-specific-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_in.csv", ",");
            Nd4j.writeNumpy(testData.getLabels(),"C:/Users/Administrator/Documents/GitHub/dl4j-examples/dl4j-cuda-specific-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_labels.csv", ",");

            System.out.println("normalize based on synthetic VR dataset...");
            normalizer.transform(huntData);     //Apply normalization to the training data

            //dummy features in cols 0 thru ?
            System.out.println("Generate post normalized dummy features 1 thru " + (num_features-1) + "...");
            huntData.getFeatures().get(NDArrayIndex.interval(0,huntData.getFeatures().rows()),NDArrayIndex.interval(0,(num_features-1))).assign(Nd4j.rand(huntData.getFeatures().rows(), (num_features-1)).sub(.5).mul(20));
            //*/
            //--------------------------------
            System.out.println("Actual data(UnwrapActualData2b):"); // lighting(Outdoor (m)?), Blind1 (i), Blind2 (j), Occupancy (k), Leaving (l)

            labelIndex = 4;
            DataSet actualData = readCSVDataset(
                "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/UnwrapActualData2b.csv",
                batchSizeTraining, labelIndex, numClasses);

            System.out.println("remove feature 3(OutdoorCondition)..."); // Occupancy,Leaving,OutdoorCondition,WorkingAreilluminance(HUNT)
            tmp = Nd4j.concat(1, actualData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(0,2)), actualData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(3,4)));
            actualData.setFeatures(tmp.dup());

            SplitTestAndTrain split3 = actualData.splitTestAndTrain(.00); // 0/84 train
            DataSet testData3 = split3.getTest(); // this is the data we want to classify

            normalizer.transform(actualData);     //Apply normalization to the training data and test data. This is using statistics calculated from the *training* set

            //--------------------------------
            /*DataSet combinedData = syntheticVRData;
            System.out.println("combine features...");
            combinedData.setFeatures(Nd4j.concat(0, trainingData_syntheticVR.getFeatures(), trainingData.getFeatures()));
            combinedData.setLabels(Nd4j.concat(0, trainingData_syntheticVR.getLabels(), trainingData.getLabels()));

            combinedData.shuffle();//*/
            /*SplitTestAndTrain split_actual = actualData.splitTestAndTrain(.00);
            DataSet trainingData_actual = split_actual.getTrain();
            DataSet testData_actual = split_actual.getTest(); // this is the data we want to classify*/

            // make the data model for records prior to normalization, because it
            // changes the data.
            //Map<Integer,Map<String,Object>> rooms = makeRoomsForTesting(testData);

            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            //normalizer.transform(syntheticHuntData);     //Apply normalization to the training data
            //normalizer.transform(actual);     //Apply normalization to the training data

            Evaluation eval;
            INDArray output;

            ////evaluate the model on the test set
            //System.out.println("\nTesting on file synthetic_hunt_data_no_hidden.csv :");
            //eval = new Evaluation(numClasses);
            //output = model.output(testData_syntheticHunt.getFeatureMatrix());
            //eval.eval(testData_syntheticHunt.getLabels(), output);
            //log.info(eval.stats());

            int len = epochs / (iterations);
            double[] vr_result = new double[len];
            double[] vr_on_correct = new double[len];
            double[] vr_off_correct = new double[len];
            double[] actual_result = new double[len];

            model.setListeners(new ScoreIterationListener(1));
            for (int n = 0; n < epochs / (iterations); n++) {
                //evaluate the model on the test set
                System.out.println("\nTesting on file synthetic_VR_data_lum.csv :");
                eval = new Evaluation(numClasses);
                output = model.output(testData_syntheticVR.getFeatureMatrix());
                eval.eval(testData_syntheticVR.getLabels(), output);
                log.info(eval.stats());
                vr_result[n]=eval.accuracy();
                double all_positives=eval.truePositives().get(1)+eval.falsePositives().get(1);
                double all_negatives=eval.trueNegatives().get(1)+eval.falseNegatives().get(1);
                if(all_positives>0) {vr_on_correct[n]=eval.truePositives().get(1)/all_positives;}
                if(all_negatives>0) {vr_off_correct[n]=eval.trueNegatives().get(1)/all_negatives;}

                //evaluate the model on the test set
                System.out.println("\nTesting on file UnwrapActualData2b.csv :");
                eval = new Evaluation(numClasses);
                output = model.output(testData3.getFeatureMatrix());
                eval.eval(testData3.getLabels(), output);
                log.info(eval.stats());
                //actual_result[n]=eval.accuracy();
                actual_result[n]=eval.accuracy();

                //model.fit(trainingData_syntheticVR);
                model.fit(trainingData_syntheticVR);
            }

            System.out.println("\nTesting on file synthetic_VR_data_lum.csv :");
            eval = new Evaluation(numClasses);
            output = model.output(testData_syntheticVR.getFeatureMatrix());
            eval.eval(testData_syntheticVR.getLabels(), output);
            log.info(eval.stats());
            Nd4j.writeNumpy(Nd4j.getExecutioner().exec(new IAMax(output),1),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum_mixed_model_out.csv", ",");

            //--------------------------------
            ////evaluate the model on the test set
            //System.out.println("\nTesting on file synthetic_hunt_data_no_hidden.csv :");
            //eval = new Evaluation(numClasses);
            //output = model.output(testData_syntheticHunt.getFeatureMatrix());
            //eval.eval(testData_syntheticHunt.getLabels(), output);
            //log.info(eval.stats());
            //Nd4j.writeNumpy(Nd4j.getExecutioner().exec(new IAMax(output),1),"C:/Users/Administrator/Documents/GitHub/dl4j-examples/dl4j-cuda-specific-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_no_hidden_mixed_model_out.csv", ",");

        } catch (Exception e){
            e.printStackTrace();
        }

        System.out.println("main ended");
    }

    //----------------------------------------------------------------------------------------------------------------------
    /**
     * take the dataset loaded for the matric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     *
     //* @param testData
     * @return
     */
    /*
    public static Map<Integer,Map<String,Object>> makeRoomsForTesting(DataSet testData){
        Map<Integer,Map<String,Object>> rooms = new HashMap<>();

        INDArray features = testData.getFeatureMatrix();
        for (int i = 0; i < features.rows() ; i++) {
            INDArray slice = features.slice(i);
            Map<String,Object> room = new HashMap();

            //set the attributes
            room.put("time", time.get(slice.getInt(0)));
            room.put("outdoor_condition", outdoor_condition.get(slice.getInt(1)));
            room.put("length_of_leave", length_of_leave.get(slice.getInt(2)));
            room.put("occupancy", occupancy.get(slice.getInt(3)));
            room.put("previous_light_state", previous_light_state.get(slice.getInt(4)));

            rooms.put(i,room);
        }
        return rooms;

    }
    */
    //----------------------------------------------------------------------------------------------------------------------
    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        try{
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
            Map<Integer,String> enums = new HashMap<>();
            for(String line:lines){
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]),parts[1]);
            }
            return enums;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }

    }

    //----------------------------------------------------------------------------------------------------------------------
    /**
     * used for testing and training
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet readCSVDataset(
        String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
        throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        return iterator.next();
    }

    //----------------------------------------------------------------------------------------------------------------------
}

