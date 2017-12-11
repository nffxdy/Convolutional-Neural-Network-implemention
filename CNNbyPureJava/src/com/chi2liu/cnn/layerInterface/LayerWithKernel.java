/**
 * @projectName: CNNbyPureJava
 * @packageName: com.chi2liu.cnn.layerInterface
 * @fileName: LayerWithKernel.java
 */
package com.chi2liu.cnn.layerInterface;

/**
 * @author: chi633 
 * @Email: cliu_whu@yeah.net
 * @date：2017年12月10日 下午4:00:38
 * @className: LayerWithKernel.java
 * @version: 1.0
 */
public interface LayerWithKernel extends Layer {
	
	/**
	 * 获取一个kernel
	 * 这个kernel的作用是和前一层的某个输出preLayerOutIndex
	 * 进行计算后
	 * 得到本层的某个输出thisLayerOutIndex
	 * @param preLayerOutIndex
	 * @param thisLayerOutIndex
	 * @return
	 * @date: 2017年12月10日
	 */
	double[][] getKernel(int preLayerOutIndex, int thisLayerOutIndex);

    void updateKernels();

    void updateBias();

}
