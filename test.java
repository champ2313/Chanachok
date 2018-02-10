package org.deeplearning4j.examples.test;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModelHelper;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Administrator on 4/19/2017.
 */

public class test {
    private static Logger log = LoggerFactory.getLogger(test.class);

    public static void main(String[] args) throws Exception {
        System.out.println("main started");
        //boolean gpu_used = Nd4j.create(1).getClass().equals(org.nd4j.linalg.jcublas.JCublasNDArray.class);
        //if (gpu_used) System.out.println("GPU use detected.");
        //else System.out.println("WARNING: no GPU use detected, base class being used is " + Nd4j.create(1).getClass());

        //import org.deeplearning4j.transferlearning.vgg16 and print summary
        TrainedModelHelper modelImportHelper = new TrainedModelHelper(TrainedModels.VGG16);
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ComputationGraph vgg16 = modelImportHelper.loadModel();
        log.info(vgg16.summary());
        String filename = "vgg16.dat";
        log.info("\n\nSaving vgg16 to " + filename + "...\n\n");
        ModelSerializer.writeModel(vgg16 , filename,true);

        System.out.println("main ended");
    }

}
