# 免费数据和模拟盘

选择 **Alpaca Basic + Paper Only**。官方说明允许全球用户以邮箱注册 Paper Only，
模拟账户免费；免费美股/ETF 实时数据仅 IEX，一个交易所，不是全市场 SIP。
本机已于 2026-09-19 测试接口可访问，无凭据返回 HTTP 401。
尚未认证成功，也没有启动模拟盘。需要用户注册免费账户，在本地环境变量配置
`APCA_API_KEY_ID` 和 `APCA_API_SECRET_KEY`，不要提交到 GitHub 或聊天。
代码不自动读取 `.env` 文件。没有购买付费数据。

```bash
pairs-trader download-alpaca --symbols EWA EWC XLP VDC XLV VHT XLI VIS --start 2016-01-01 --end 2026-09-19 --feed iex --output output/alpaca/research
```

命令使用全新目录、纽约时间的排他结束日、完整分页，分别保存 raw / all-adjusted
行情及元数据。遇到认证或权限错误直接失败，不混用行情源。
这些文件是研究 bars，尚未整合股息事件，所以不能直接替换美股现金流账本输入。
现有 paper 命令已显式使用 IEX、复权研究价格和上一个已结束日之前的数据；
它仍是原 Kalman 策略，与多交易对动态研究尚未连成自动下单系统。

当前多交易对回测使用 Yahoo/yfinance 免费历史行情，并在所有报告标明来源。
Yahoo 是非官方研究接口；可用于推进研究，不能称作新的机构级行情源。
同一策略最终需要使用拟交易 feed 重新校准及验证。没有免费的日线 API 能验证
订单排队、真实盘口冲击及高频交易表现。

其他候选：Twelve Data Basic 免费但需要 key，官网列出每日 800 次请求；
Alpha Vantage 免费通常每天 25 次，历史与高级端点有额外限制。
本机 Stooq CSV 测试返回 HTTP 404，未采用。不存在“已切换到 Alpaca 数据”的结论。

官方来源：
- [Alpaca Paper Only 注册与模拟局限](https://docs.alpaca.markets/us/docs/paper-trading)
- [Alpaca Basic 数据覆盖和认证](https://docs.alpaca.markets/us/docs/about-market-data-api)
- [Bars 分页及调整字段](https://docs.alpaca.markets/us/reference/stockbars)
- [Twelve Data 免费方案](https://twelvedata.com/stocks)
- [Alpha Vantage 免费额度](https://www.alphavantage.co/support/)
