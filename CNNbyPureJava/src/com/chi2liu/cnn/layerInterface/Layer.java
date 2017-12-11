/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerInterface
 * @fileName: Layer.java
 */
package com.chi2liu.cnn.layerInterface;

import java.util.function.Consumer;
import java.util.function.Function;

import com.chi2liu.cnn.data.DataSet;
import com.chi2liu.cnn.util.Size;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date��2017��12��10�� ����3:50:08
 * @className: Layer.java
 * @version: 1.0
 */
public interface Layer {
	
	/**
	 * ��ȡ�˲��Size��С
	 * @return
	 * @date: 2017��12��10��
	 */
	Size getSize();
	
	/**
	 * ��ȡ�˲�����map�ĸ���
	 * @return
	 * @date: 2017��12��10��
	 */
	int getOutCount();
	
	
	/**
	 * �Դ˲���ƶ���һ����¼��ÿһ��������в���
	 * ����Ĳ��������ں�����
	 * @param recordIndex
	 * @param function
	 * @date: 2017��12��10��
	 */
	void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function);
	
	/**
	 * ������֮��������һ��
	 * @param layer
	 * @date: 2017��12��10��
	 */
	void setNextLayer(Layer layer);
	
	/**
	 * ��ȡָ����¼��ָ����һ�������errors
	 * @param recordIndex
	 * @param outIndex
	 * @return
	 * @date: 2017��12��10��
	 */
	double[][] getError(int recordIndex, Integer outIndex);
	
	/**
	 * ��ȡָ����¼��ָ����һ������Ľ��
	 * @param recordIndex
	 * @param outIndex
	 * @return
	 * @date: 2017��12��10��
	 */
	double[][] getOut(int recordIndex, int outIndex);
	
	/**
	 * ����һ����¼����ǰ�򴫲�
	 * @param record
	 * @date: 2017��12��10��
	 */
	void forward(DataSet.Record record);
	
	/**
	 * ����һ����¼���з��򴫲�
	 * @param record
	 * @return
	 * @date: 2017��12��10��
	 */
	boolean backPropagation(DataSet.Record record);

}
