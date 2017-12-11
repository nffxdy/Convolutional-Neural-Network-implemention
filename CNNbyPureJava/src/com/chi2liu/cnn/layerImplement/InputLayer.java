/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerImplement
 * @fileName: InputLayer.java
 */
package com.chi2liu.cnn.layerImplement;

import java.util.function.Consumer;
import java.util.function.Function;

import com.chi2liu.cnn.data.DataSet.Record;
import com.chi2liu.cnn.layerInterface.Layer;
import com.chi2liu.cnn.util.Size;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date��2017��12��10�� ����4:03:23
 * @className: InputLayer.java
 * @version: 1.0
 */
/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date��2017��12��10�� ����4:16:00
 * @className: InputLayer.java
 * @version: 1.0
 */
public class InputLayer implements Layer {
	
	
	/**
	 * ÿ����һ��Size
	 * ָ�������map�ĳߴ��С
	 */
	private final Size size;
	/**
     * ÿ������Ϊһ������
     */
    private final double[][][] outs;	
	
	public InputLayer(int batchSize, Size size) {
        this.size = size;
        this.outs = new double[batchSize][size.getWidth()][size.getHeight()];
    }
	
	/**
	 * ������������ǽ�record�ļ�¼��ȡ����ת���ɶ�Ӧsize�ľ���
	 * @param index
	 * @param values
	 * @date: 2017��12��10��
	 */
	private void setOutput(int index, double[] values) {
        for (int x = 0; x < size.getWidth(); x++) {
            for (int y = 0; y < size.getHeight(); y++) {
                outs[index][x][y] = values[size.getWidth() * x + y];
            }
        }
    }

	@Override
	public Size getSize() {
		return size;
	}

	/**
	 * ���������ֻ��һ��
	 * @see com.chi2liu.cnn.layerInterface.Layer#getOutCount()
	 */
	@Override
	public int getOutCount() {
		return 1;
	}

	@Override
	public void forEachOutput(int recordIndex, Function<Integer, Consumer<double[][]>> function) {
		function.apply(0).accept(outs[recordIndex]);
	}

	@Override
	public void setNextLayer(Layer layer) {
	}

	/**
	 * �����û��errors
	 * @see com.chi2liu.cnn.layerInterface.Layer#getError(int, java.lang.Integer)
	 */
	@Override
	public double[][] getError(int recordIndex, Integer outIndex) {
		throw new RuntimeException("no error for input layer");
	}

	/**
	 * ��Ϊ�����ֻ��һ�����
	 * ����outindex�����ò���
	 * ֻ��Ϊ�˱��ֽӿڵ�һ���ԣ��������ﻹ���������
	 * @see com.chi2liu.cnn.layerInterface.Layer#getOut(int, int)
	 */
	@Override
	public double[][] getOut(int recordIndex, int outIndex) {
		return outs[recordIndex];
	}

	/**
	 * ������ǰ�򴫲�
	 * �������ú����������
	 * ����record�ļ�¼ת����һ������
	 * @see com.chi2liu.cnn.layerInterface.Layer#forward(com.chi2liu.cnn.data.DataSet.Record)
	 */
	@Override
	public void forward(Record record) {
		setOutput(record.getIndex(), record.getData());
	}

	@Override
	public boolean backPropagation(Record record) {
		return setErrors(record.getIndex(), record);
	}

	private boolean setErrors(int index, Record record) {
		return true;
	}

}
