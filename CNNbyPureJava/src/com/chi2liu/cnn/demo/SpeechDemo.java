package com.chi2liu.cnn.demo;

import java.io.IOException;
import java.util.logging.Logger;

import javax.xml.parsers.ParserConfigurationException;

import com.chi2liu.cnn.cnn.CNN;
import com.chi2liu.cnn.cnn.CNNBuilder;
import com.chi2liu.cnn.data.DataSet;
import com.chi2liu.cnn.util.Size;


public class SpeechDemo {
	
	private static Logger logger = Logger.getLogger(Demo.class.getSimpleName());

    public static void main(String[] args) throws IOException, ParserConfigurationException, ClassNotFoundException {
        DataSet dataSet = new DataSet("speech/train.csv", 0.05);
        System.out.println(dataSet.getTrainSize());
        CNN cnn = new CNNBuilder(180)
                .setInputLayer(new Size(6, 6))
                .addConvolutionalLayer(4, new Size(3, 3))
                .addSimpleLayer(new Size(2, 2))
                .setOutputLayer(2)
                .build();
        long now = System.currentTimeMillis();
        cnn.train(dataSet, 3000);
        System.out.println("cost:" + (System.currentTimeMillis() - now));
        cnn.saveModel("demo.model");


        CNN cnn1 = CNN.readModel("demo.model");
        final int[] testRight = {0};
        final int[] testCount = {0};
        dataSet.testRecordForEach(record -> {
            if (cnn1.test(record)) {
                testRight[0]++;
            }
            testCount[0]++;
        });
        double testP = 1.0 * testRight[0] / testCount[0];
        logger.info("test precision " + testRight[0] + "/" + testCount[0] + "=" + testP);
    }

}
