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
 * @date：2017年12月10日 下午4:03:23
 * @className: InputLayer.java
 * @version: 1.0
 */
/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午4:16:00
 * @className: InputLayer.java
 * @version: 1.0
 */
public class InputLayer implements Layer {
	
	
	/**
	 * 每层有一个Size
	 * 指定本层的map的尺寸大小
	 */
	private final Size size;
	/**
     * 每个输入为一个矩形
     */
    private final double[][][] outs;	
	
	public InputLayer(int batchSize, Size size) {
        this.size = size;
        this.outs = new double[batchSize][size.getWidth()][size.getHeight()];
    }
	
	/**
	 * 输入层的输出就是将record的记录读取，并转换成对应size的矩形
	 * @param index
	 * @param values
	 * @date: 2017年12月10日
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
	 * 输入层的输出只有一个
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
	 * 输入层没有errors
	 * @see com.chi2liu.cnn.layerInterface.Layer#getError(int, java.lang.Integer)
	 */
	@Override
	public double[][] getError(int recordIndex, Integer outIndex) {
		throw new RuntimeException("no error for input layer");
	}

	/**
	 * 因为输入层只有一个输出
	 * 所以outindex参数用不到
	 * 只是为了保持接口的一致性，所以这里还有这个参数
	 * @see com.chi2liu.cnn.layerInterface.Layer#getOut(int, int)
	 */
	@Override
	public double[][] getOut(int recordIndex, int outIndex) {
		return outs[recordIndex];
	}

	/**
	 * 输入层的前向传播
	 * 就是设置好输入层的输出
	 * 即将record的记录转换成一个矩形
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
