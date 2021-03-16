x = (6:40)
Vm = log((5+0.03*120+0.1*5)/(125+0.03*12+0.1*125)) *26.7*(x+273)/310
p = plot(x,Vm,'-o','MarkerSize',6,'LineWidth',2)
xlabel('温度')
ylabel('膜电位')
title('膜电位与温度的关系')
