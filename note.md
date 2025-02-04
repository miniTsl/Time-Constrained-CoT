不同类型的模型（vendor, reasoning，math，general）

相同模型，不同大小

不同prompt类型

不同token budget

不同数据集

只改变1个因素
1. 固定数据集、模型类型、prompt类型、token budget，考察不同模型大小：正常结论是模型越大，准确率越高
2. 固定数据集、模型类型、prompt类型、模型大小，考察不同token budget：正常结论是token budget越大，准确率越高（显然，修改last_number）
3. 固定数据集、模型类型、token budget、模型大小，考察不同prompt类型：正常结论是直接截断效果不如加截止符（1图说明），c2f和aav在某些模型上也会比sbs要好
4. 固定数据集、token budget、prompt类型、模型大小，考察不同模型类型：
    a) 普通的和o-1的：正常结论是o-1的准确率更高
    b) 普通的和数学的：正常结论是数学的准确率更高
    c) 数学的和o-1的：正常结论是o-1的准确率更高
    d) vendor?
5. 固定模型类型、token budget、prompt类型、模型大小，考察不同数据集：正常结论是数据集越简单，在有budget限制下，acc上升越快，最终的收敛acc也越高

如果换算到真实的latency...

改变2个因素
