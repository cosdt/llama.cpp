### 安装CANN-nnal(需与toolkit版本一致)

默认的安装位置为`Ascend/nnal/atb`

```bash
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-aarch64.run && \ 
chmod +x Ascend-cann-nnal_8.1.RC1_linux-aarch64.run && \
./Ascend-cann-nnal_8.1.RC1_linux-aarch64.run --install -q && \
source ~/Ascend/nnal/atb/set_env.sh
```

### ATB基础定义

#### TensorDesc

包含对Tensor的相关描述信息：每个Tensor的数据类型，数据排布格式和形状维度信息。

```
struct TensorDesc {
    aclDataType dtype = ACL_DT_UNDEFINED; // Tensor数据类型。
    aclFormat format = ACL_FORMAT_UNDEFINED; //Tensor数据排布格式。
    Dims shape; // Tensor数据的形状，
};
```

#### Dims

Shape维度信息。

```
struct Dims {
    int64_t dims[MAX_DIM]; // 每一维的大小，要求大于0。
    uint64_t dimNum = 0; // Tensor的维数，取值范围为(0, 8]。
};
```



#### Tensor

包含每个Tensor的描述符、NPU内存地址、CPU内存地址和内存大小等。

```
struct Tensor {
    TensorDesc desc; // Tensor描述信息。
    void *deviceData = nullptr; // TensorNPU内存地址。
    void *hostData = nullptr; // TensorCPU内存地址。
    uint64_t dataSize = 0; // “deviceData”或“hostData”指向内容的内存大小。
};
```

#### VariantPack

加速库算子执行时需要构造VariantPack存放输入及最终输出。

```
struct VariantPack {
    SVector<Tensor> inTensors; // 存放所有输入tensor的SVector。
    SVector<Tensor> outTensors;// 存放所有输出tensor的SVector。
};
```



### ATB算子ADD使用示例

#### 代码

```c++
#include <iostream>
#include <vector>
#include <acl/acl.h>
#include <atb/types.h>
#include <atb/atb_infer.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"


void CreateInTensorDescs(atb::SVector<atb::TensorDesc> &intensorDescs) 
{
    for (size_t i = 0; i < intensorDescs.size(); i++) {
        intensorDescs.at(i).dtype = ACL_FLOAT16;
        intensorDescs.at(i).format = ACL_FORMAT_ND;
        intensorDescs.at(i).shape.dimNum = 2;
        intensorDescs.at(i).shape.dims[0] = 2;
        intensorDescs.at(i).shape.dims[1] = 2;
    }
}

// 设置intensor，并分配、初始化数据为0
void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &intensorDescs)
{
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = intensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST); 
        if (ret != 0) {
            std::cout << "alloc error1!";
            exit(0);
        }
        ret = aclrtMemset(inTensors[i].deviceData, inTensors[i].dataSize, 0, inTensors[i].dataSize);
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
}

// 设置outtensor并分配内存
void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outtensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outtensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }
}

int main() {
    // 配置deviceId
    uint32_t deviceId = 0;
    aclError status = aclrtSetDevice(deviceId);

    // 创建算子对象实例，以ADD算子为例
    // 构造Operation参数
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;


    // 创建算子对象实例
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation(addParam, &op);


    // 创建输入输出tensor，存入VariantPack
    atb::VariantPack pack;
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;

    uint32_t inTensorNum = op->GetInputNum();
    pack.inTensors.resize(inTensorNum);
    intensorDescs.resize(inTensorNum);
    // 创建tensor描述
    CreateInTensorDescs(intensorDescs); 
    // 根据tensor描述创建tensor
    CreateInTensors(pack.inTensors, intensorDescs);

    uint32_t outTensorNum = op->GetOutputNum();
    outtensorDescs.resize(outTensorNum);
    pack.outTensors.resize(outTensorNum);
    // 以输入tensor描述推导输出tensor描述,创建输出tensor
    op->InferShape(intensorDescs, outtensorDescs);
    CreateOutTensors(pack.outTensors, outtensorDescs);


    // 创建context，配置stream
    atb::Context *context = nullptr;
    st = atb::CreateContext(&context);

    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);
    context->SetExecuteStream(stream);

    // 调用Setup接口，计算workspace大小
    uint64_t workspaceSize = 0;
    st = op->Setup(pack, workspaceSize, context);

    // 根据workspace大小申请NPU内存
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        aclError status = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != 0) {
            std::cout << "alloc error!";
            exit(0);
        }
    }

    // 调用Execute接口，执行算子
    st = op->Execute(pack, (uint8_t *)workspace, workspaceSize, context);

    // 销毁创建的对象，释放内存
    // 流同步等待device侧任务计算完成
    auto ret = aclrtSynchronizeStream(stream);
    if (ret != 0) {
        std::cout << "sync error!";
        exit(0);
    }

    status = aclrtDestroyStream(stream); // 销毁stream
    st = atb::DestroyOperation(op);      // 销毁op对象
    st = atb::DestroyContext(context);   // 销毁context

    for (size_t i = 0; i < pack.inTensors.size(); i++) {
        aclrtFree(pack.inTensors.at(i).deviceData);
    }

    for (size_t i = 0; i < pack.outTensors.size(); i++) {
        aclrtFree(pack.outTensors.at(i).deviceData);
    }
    status = aclrtFree(workspace);       // 销毁workspace
    aclrtResetDevice(deviceId);          // 重置deviceId

    return 0;
}
```

#### 编译命令

```bash
g++ -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" atb_demo.cpp -l atb -l ascendcl -o atb_demo

./atb_demo
```

#### Note:

```
// 单独创建 atb::TensorDesc 示例
   	atb::TensorDesc desc;
    desc.dtype = ACL_FLOAT16;
    desc.format = ACL_FORMAT_ND;
    desc.shape.dimNum = 2;
    desc.shape.dims[0] = 2;
    desc.shape.dims[1] = 2;
    intensorDescs.push_back(desc);
    
    
// 单独创建 atb::Tensor 示例
    atb::Tensor tensor;
    tensor.desc = intensorDescs.at(0);
    tensor.dataSize = atb::Utils::GetTensorSize(tensor);
    int ret = aclrtMalloc(&tensor.deviceData, tensor.dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0) {
        std::cout << "alloc error1!";
        exit(0);
    }
    ret = aclrtMemset(tensor.deviceData, tensor.dataSize, 0, tensor.dataSize);
    if (ret != 0) {
        std::cout << "alloc error2!";
        exit(0);
    }
    inTensors.push_back(tensor);
```



示例使用了`atb::SVector`中**两个函数**：

- **`push_back()`:** 向容器内添加元素

-  **`resize()`**：设定容器大小，但同时内部会初始化存储对象，不需要`push_back`在进行添加



### llama.cpp应用示例

**修改ggml-cann/CMakeLists.txt 引入atb头文件、so文件 **

```
    list(APPEND CANN_LIBRARIES
        ascendcl
        nnopbase
        opapi
        acl_op_compiler
        atb
    )    
    target_include_directories(ggml-cann PRIVATE ${CANN_INCLUDE_DIRS} $ENV{ATB_HOME_PATH}/include)
    target_link_directories(ggml-cann PRIVATE ${CANN_INSTALL_DIR}/lib64 $ENV{ATB_HOME_PATH}/lib)
    
```

代码示例：

主要有以下几点修改：

- 输入tensorDesc 的数据类型及dims
- 输入输出tensor的  deviceData数据地址
- stream使用ctx默认流

```c++
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        atb::Operation *op = nullptr;
        atb::Status st = atb::CreateOperation(addParam, &op);
        // 创建输入输出tensor，存入VariantPack
        atb::VariantPack pack;
        // 构建输入tensor
        uint32_t inTensorNum = op->GetInputNum();
        atb::SVector<atb::TensorDesc> intensorDescs;

        pack.inTensors.resize(inTensorNum);
        intensorDescs.resize(inTensorNum);

        for (size_t i = 0; i < intensorDescs.size(); i++) {
            if(src0->type == GGML_TYPE_F16){
                intensorDescs.at(i).dtype = ACL_FLOAT16;
            }else{
                intensorDescs.at(i).dtype = ACL_FLOAT;
            }
            
            intensorDescs.at(i).format = ACL_FORMAT_ND;
            intensorDescs.at(i).shape.dimNum = GGML_MAX_DIMS;
        }

        intensorDescs.at(0).shape.dims[0] = src0->ne[3];
        intensorDescs.at(0).shape.dims[1] = src0->ne[2];
        intensorDescs.at(0).shape.dims[2] = src0->ne[1];
        intensorDescs.at(0).shape.dims[3] = src0->ne[0];

        intensorDescs.at(1).shape.dims[0] = src1->ne[3];
        intensorDescs.at(1).shape.dims[1] = src1->ne[2];
        intensorDescs.at(1).shape.dims[2] = src1->ne[1];
        intensorDescs.at(1).shape.dims[3] = src1->ne[0];

        for (size_t i = 0; i < pack.inTensors.size(); i++) {
            pack.inTensors.at(i).desc = intensorDescs.at(i);
            pack.inTensors.at(i).dataSize = atb::Utils::GetTensorSize(pack.inTensors.at(i));
        }
        pack.inTensors.at(0).deviceData = src0->data;
        pack.inTensors.at(1).deviceData = src1->data;

        // 构建输出tensor
        uint32_t outTensorNum = op->GetOutputNum();
        atb::SVector<atb::TensorDesc> outtensorDescs;
        outtensorDescs.resize(outTensorNum);
        pack.outTensors.resize(outTensorNum);
        op->InferShape(intensorDescs, outtensorDescs);

        for (size_t i = 0; i < pack.outTensors.size(); i++) {
            pack.outTensors.at(i).desc = outtensorDescs.at(i);
            pack.outTensors.at(i).dataSize = atb::Utils::GetTensorSize(pack.outTensors.at(i));
            pack.outTensors.at(i).deviceData = dst->data;
        }

        atb::Context *context = nullptr;
        atb::CreateContext(&context);
        context->SetExecuteStream(ctx.stream());

        uint64_t workspaceSize = 0;
        op->Setup(pack, workspaceSize, context);

        void *workspace = nullptr;
        if (workspaceSize != 0) {
            aclError status = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (status != 0) {
                std::cout << "alloc error!";
                exit(0);
            }
        }
        op->Execute(pack, (uint8_t *)workspace, workspaceSize, context);
        st = atb::DestroyOperation(op);      // 销毁op对象
        st = atb::DestroyContext(context);   // 销毁context
    
        aclrtFree(workspace); 
```



