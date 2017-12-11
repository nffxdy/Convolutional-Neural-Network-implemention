/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.demo
 * @fileName: Demo.java
 */
package com.chi2liu.cnn.demo;

import java.io.IOException;
import java.util.logging.Logger;

import javax.xml.parsers.ParserConfigurationException;

import com.chi2liu.cnn.cnn.CNN;
import com.chi2liu.cnn.cnn.CNNBuilder;
import com.chi2liu.cnn.data.DataSet;
import com.chi2liu.cnn.util.Size;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午6:49:09
 * @className: Demo.java
 * @version: 1.0
 */
public class Demo {

	private static Logger logger = Logger.getLogger(Demo.class.getSimpleName());

    public static void main(String[] args) throws IOException, ParserConfigurationException, ClassNotFoundException {
        DataSet dataSet = new DataSet("dataSet/data.ds", 0.3);
        System.out.println(dataSet.getTrainSize());
        CNN cnn = new CNNBuilder(10)
                .setInputLayer(new Size(28, 28))
                .addConvolutionalLayer(6, new Size(5, 5))
                .addSimpleLayer(new Size(2, 2))
                .addConvolutionalLayer(12, new Size(5, 5))
                .addSimpleLayer(new Size(2, 2))
                .setOutputLayer(10)
                .build();
        long now = System.currentTimeMillis();
        cnn.train(dataSet, 100);
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
