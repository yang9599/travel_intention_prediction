### 数据下载
通过百度云链接下载：https://pan.baidu.com/s/1_BQWTSLI0mMpTE87pHIyZA

### 数据表介绍
轨迹数据表

| 字段名称      | 中文解释    | 备注说明  |
| --------     | -----      |---------|
| user_id      |   用户标识  | 抽样&脱敏 |
| longitude    | 经度        | 保留小数点后三位 |
| latitude     | 维度        | 保留小数点后三位 |
| start_time   |数据采集时间  |datetime格式|

用户出行行为数据表

| 字段名称      | 中文解释         | 备注说明  |
| --------     | -----            |---------|
| user_id      |   用户标识        |  抽样&脱敏 |
| flag         | 是否有旅行出行需求 |	0：无出行意向；   1：有出行意向 |
| travel_type  | 旅游类型          |	当flag=0，travel_type=0；当flag=1，travel_type=1表示境外游、travel_type=2表示省外游、travel_type=3表示省内游 |
