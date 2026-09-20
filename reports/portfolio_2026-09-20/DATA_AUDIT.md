# 数据采集与覆盖审计

本次实际认证下载 Alpaca IEX，23 个代码，原始日线 35,322 条，另有同范围 ALL-adjusted 日线。另采集 Yahoo 55,776 条作为冻结协议的主研究数据。

Alpaca 请求 2017–2026，但多数代码实际从 2020-07-27 才开始，SPY 有一条更早的孤立记录。未补造缺失数据，未将短历史声称为完整 2018–2022 训练集。IEX 仅单一交易所；流动性阈值和容量数值均为该 feed 代理，不可解释为全市场容量。

同日期、同代码可比的相邻共同观测收益中，有 897 个差异超过 1 个百分点。见 source_return_discrepancies.csv，差异包含 feed 收盘、缺失 bar 和复权因素，未据此自动更正任一来源。跨来源差异意味着不能将一个来源的历史模型表现视为另一个来源的可执行收益。

原始文件：output/portfolio_v1/alpaca_data/bars_raw.csv 与 bars_all.csv；Yahoo 原始文件及元数据在 output/portfolio_v1/market_data。各文件 SHA256 见 data_file_hashes.json，逐代码起止日期见 source_coverage.csv。密钥未写入项目文件；仅经进程环境用于只读下载，没有调用下单接口。
