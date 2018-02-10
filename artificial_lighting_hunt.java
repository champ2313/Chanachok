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
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;

/**
 * Created by Administrator on 5/2/2017.
 * Based on 'BasicCSVClassifier'
 */
public class artificial_lighting_hunt {
    private static Logger log = LoggerFactory.getLogger(test.class);

    //private static Map<Integer,String> time = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/time");
    //private static Map<Integer,String> outdoor_condition = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/outdoor_condition.csv");
    //private static Map<Integer,String> length_of_leave = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/length_of_leave.csv");
    //private static Map<Integer,String> occupancy = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/occupancy.csv");
    //private static Map<Integer,String> previous_light_state = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/previous_light_state.csv");
    //private static Map<Integer,String> light_state = readEnumCSV("C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/light_state.csv");

    //----------------------------------------------------------------------------------------------------------------------
    public static void main(String[] args) throws Exception {
        System.out.println("main started");
        //boolean gpu_used = Nd4j.create(1).getClass().equals(org.nd4j.linalg.jcublas.JCublasNDArray.class);
        //if (gpu_used) System.out.println("GPU use detected.");
        //else System.out.println("WARNING: no GPU use detected, base class being used is " + Nd4j.create(1).getClass());

        try {
            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            int num_features=4; // Occupancy, Leaving, OutdoorCondition, WorkingAreilluminance(HUNT)
            int labelIndex = num_features;     //5 values in each row of synthetic_VR_data_lum.csv: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            int numClasses = 2;     //2 classes (light off/light on). Classes have integer values 0 or 1
            int epochs = 1000; // 100
            int batchSizeTraining = 100000;    //artificial lighting Hunt data set: 100,000 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
            INDArray tmp;

            //--------------------------------
            System.out.println("Read synthetic VR dataset(synthetic_VR_data_lum)...");

            labelIndex = 4;
            DataSet vrData = readCSVDataset(
                "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum.csv", // Occupancy,Leaving,OutdoorCondition,WorkingAreilluminance(HUNT)
                batchSizeTraining, labelIndex, numClasses, false);

            System.out.println("remove feature 3(OutdoorCondition)..."); // Occupancy,Leaving,WorkingAreilluminance(HUNT)
            tmp = Nd4j.concat(1, vrData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(0,2)), vrData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(3,4)));
            vrData.setFeatures(tmp.dup());
            num_features = num_features - 1;//*/

            SplitTestAndTrain split2 = vrData.splitTestAndTrain(.20);
            DataSet trainingData2 = split2.getTrain();
            DataSet testData2 = split2.getTest(); // this is the data we want to classify
            Nd4j.writeNumpy(testData2.getFeatureMatrix(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum_in.csv", ",");
            Nd4j.writeNumpy(testData2.getLabels(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum_labels.csv", ",");

            System.out.println("Make normalizer based on synthetic VR dataset...");
            //System.out.println("Data Order: Blind1, Blind2, Occupancy, Leaving, Outdoor_light"); // Woking_Area_Light
            System.out.println("Data Order: Occupancy, Leaving, WorkingAreilluminance(HUNT)"); // Woking_Area_Light
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData2);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data

            //--------------------------------
            System.out.println("Train on Hunt data(synthetic_hunt_data_continuous):");
            System.out.println("Read Hunt dataset...");

            labelIndex = 1;
            DataSet huntData = readCSVDataset(
                "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_blank.csv",
                batchSizeTraining, labelIndex, numClasses, false);
            DataSet huntData2 = readCSVDataset(
                "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_new.csv",
                batchSizeTraining, labelIndex, 1, true);

            int l = huntData2.getLabels().length();
            for (int i=1;i<l;i++) {
                huntData.getLabels().putScalar(new int[]{i, 1}, huntData2.getLabels().getDouble(i));
                huntData.getLabels().putScalar(new int[]{i, 0}, 1-huntData2.getLabels().getDouble(i));
            }

            System.out.println("Add " + (num_features-1) + " blank features before real feature...");
            tmp = Nd4j.concat(1,Nd4j.zeros(huntData.getFeatures().rows(), (num_features-1)),huntData.getFeatures());
            huntData.setFeatures(tmp.dup());

            System.out.println("Split testing and training sets...");
            SplitTestAndTrain split = huntData.splitTestAndTrain(.90);
            DataSet trainingData = split.getTrain();
            DataSet testData = split.getTest(); // this is the data we want to classify
            Nd4j.writeNumpy(testData.getFeatureMatrix(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_in.csv", ",");
            Nd4j.writeNumpy(testData.getLabels(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_labels.csv", ",");

            // make the data model for records prior to normalization, because it
            // changes the data.
            //Map<Integer,Map<String,Object>> rooms = makeRoomsForTesting(testData);

            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            System.out.println("normalize based on synthetic VR dataset...");
            normalizer.transform(huntData);     //Apply normalization to the training data and the test data. This is using statistics calculated from the *training* set

            //dummy features in cols 0 thru ?
            System.out.println("Generate post normalized dummy features 1 thru " + (num_features-1) + "...");
            huntData.getFeatures().get(NDArrayIndex.interval(0,huntData.getFeatures().rows()),NDArrayIndex.interval(0,(num_features-1))).assign(Nd4j.rand(huntData.getFeatures().rows(), (num_features-1)).sub(.5).mul(20));

            final int numInputs = num_features;
            int outputNum = numClasses;
            int iterations = 1;
            long seed = 6;

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.02)
                .regularization(true).l2(1e-3)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(30).build())
                .layer(1, new DenseLayer.Builder().nIn(30).nOut(30).build())


                //.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)

                //Create output layer
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(30)
                    .nOut(outputNum)
                    .activation(Activation.IDENTITY).nIn(30).nOut(outputNum).build())
                    //.lossFunction(LossFunctions.LossFunction.RMSE)


                    //.activation(Activation.SOFTMAX).nIn(30).nOut(outputNum).build())
                    //.activation(Activation.SOFTMAX).nIn(30).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();



            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(1));

            for (int n=0;n<epochs/iterations;n++) {
                model.fit(trainingData);
            }

            //evaluate the model on the test set
            Evaluation eval = new Evaluation(numClasses);
            INDArray output = model.output(testData.getFeatureMatrix(),false);

            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());

            Nd4j.writeNumpy(Nd4j.getExecutioner().exec(new IAMax(output),1),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_hunt_data_continuous_out.csv", ",");

            //--------------------------------
            System.out.println("Synthetic VR data(synthetic_VR_data_lum):");
            normalizer.transform(vrData);     //Apply normalization to the training data and test data. This is using statistics calculated from the *training* set

            //evaluate the model on the test set
            eval = new Evaluation(numClasses);
            output = model.output(testData2.getFeatureMatrix());

            eval.eval(testData2.getLabels(), output);
            log.info(eval.stats());
            Nd4j.writeNumpy(Nd4j.getExecutioner().exec(new IAMax(output),1),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/synthetic_VR_data_lum_out.csv", ",");

            //--------------------------------
            System.out.println("Actual data(UnwrapActualData2b):"); // lighting(Outdoor (m)?), Blind1 (i), Blind2 (j), Occupancy (k), Leaving (l)

            labelIndex = 4;
            DataSet actualData = readCSVDataset(
                "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/UnwrapActualData2b.csv",
                batchSizeTraining, labelIndex, numClasses, false);

            System.out.println("remove feature 3(OutdoorCondition)..."); // Occupancy,Leaving,OutdoorCondition,WorkingAreilluminance(HUNT)
            tmp = Nd4j.concat(1, actualData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(0,2)), actualData.getFeatures().get(NDArrayIndex.all(), NDArrayIndex.interval(3,4)));
            actualData.setFeatures(tmp.dup());

            SplitTestAndTrain split3 = actualData.splitTestAndTrain(.00);
            DataSet testData3 = split3.getTest(); // this is the data we want to classify
            Nd4j.writeNumpy(testData3.getFeatureMatrix(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/UnwrapActualData2b_in.csv", ",");
            Nd4j.writeNumpy(testData3.getLabels(),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/UnwrapActualData2b_labels.csv", ",");

            normalizer.transform(actualData);     //Apply normalization to the training data and test data. This is using statistics calculated from the *training* set

            //evaluate the model on the test set
            eval = new Evaluation(numClasses);
            output = model.output(testData3.getFeatureMatrix());

            eval.eval(testData3.getLabels(), output);
            log.info(eval.stats());
            Nd4j.writeNumpy(Nd4j.getExecutioner().exec(new IAMax(output),1),"C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting/UnwrapActualData2b_out.csv", ",");
            //*/
            //--------------------------------
            String nn_dir = "C:/Users/cchokw1/dl4j-examples/dl4j-examples/src/main/resources/DataExamples/artificial_lighting";
            String model_name = "artificial_lighting_hunt_lum";
            String nn_file_name = nn_dir + "/" + model_name + "_net.dat";
            String nn_file_name_model = nn_dir + "/" + model_name + "_net_model.dat";
            System.out.printf("Finished, network trained. Saving trained net " + nn_file_name + " and " + nn_file_name_model + "\n");


            ModelSerializer.writeModel(model, nn_file_name_model, true); // write nn
            model.clear();
            model = null; // and clear it

            FileOutputStream fileOut = new FileOutputStream(nn_file_name);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(normalizer); // write everything else
            out.close();
            fileOut.close();
            System.out.printf("Serialized model is saved in " + nn_file_name + " and " + nn_file_name_model + "\n");
            //--------------------------------
            int width = 200;
            int height = 100;
            double xmin=0, xmax=1000,ymin=-.2, ymax=1.1;
            int[][] yourmatrix = new int[width][height];
            try {
                BufferedImage image = new BufferedImage(width, height, TYPE_INT_RGB);
                for(int i=0; i<width; i++) {
                    for(int j=0; j<height; j++) {
                        double x = xmin+(xmax-xmin)*((double)i/(double)(width-1));
                        double y = ymin+(ymax-ymin)*((double)j/(double)(height-1));
                        INDArray input = Nd4j.zeros(1,3);
                        input.putScalar(0, y); // Occupancy, Leaving, WorkingAreilluminance(HUNT)
                        input.putScalar(1,0);
                        input.putScalar(2, x);
                        normalizer.transform(input);
                        output = model.output(input);
                        //int a = 0;
                        //if (output.getDouble(1)>.5){a=255;}
                        int a = (int) (255*(output.getDouble(1)+(1-output.getDouble(0)))/2);//yourmatrix[i][j];
                        if (a<0){a=0;}
                        if (a>255){a=255;}
                        Color newColor = new Color(a,a,a);
                        image.setRGB(i,height-1-j,newColor.getRGB());
                    }
                }
                double[] hunt1occ = new double[width];
                double[] hunt1occ_lum = new double[width];
                for(int i=0; i<width; i++) {
                    double x = xmin+(xmax-xmin)*((double)i/(double)(width-1));
                    double y = 1;
                    INDArray input = Nd4j.zeros(1,3);
                    input.putScalar(0, y);
                    input.putScalar(1,0);
                    input.putScalar(2, x);
                    normalizer.transform(input);
                    output = model.output(input);
                    hunt1occ[i] = (output.getDouble(1)+(1-output.getDouble(0)))/2;
                    hunt1occ_lum[i] = x;
                }
                ImageIO.write(image, "jpg", new File("Hunt.jpg"));
                System.out.println("img out");
            }

            catch(Exception e) { System.out.println("IMG FAIL");}
            //--------------------------------
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
    /*public static Map<Integer,Map<String,Object>> makeRoomsForTesting(DataSet testData){
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

    }*/

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
        String csvFileClasspath, int batchSize, int labelIndex, int numClasses, boolean regression)
        throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        //rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        if (regression) { iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses, regression); }
        return iterator.next();
    }

    //----------------------------------------------------------------------------------------------------------------------
}
