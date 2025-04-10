//+------------------------------------------------------------------+
//|                                         Sythetic EURUSDChart.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_separate_window  // Display in a seperate window
#property indicator_buffers 4        //Buffers for Open, High,Low,Close
#property indicator_plots 1           //Plot a single series(candlesticks)
//+------------------------------------------------------------------+
//| Indicator to generate and display synthetic currency data        |
//+------------------------------------------------------------------+

double openBuffer[];
double highBuffer[];
double lowBuffer[];
double closeBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//---Set buffers for synthetic data
   SetIndexBuffer(0, openBuffer);
   SetIndexBuffer(1, highBuffer);
   SetIndexBuffer(2, lowBuffer);
   SetIndexBuffer(3, closeBuffer);
//---Define the plots for candle sticks
IndicatorSetString(INDICATOR_SHORTNAME, "Synthetic Candlestick");

//---Set the plot type for the candlesticks
   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_CANDLES);
   
   //---Setcolours for the candlesticks
   PlotIndexSetInteger(0, PLOT_COLOR_INDEXES, clrGreen);
   PlotIndexSetInteger(1, PLOT_COLOR_INDEXES, clrRed);
   //---Set the width of the candlesticks
   PlotIndexSetInteger(0, PLOT_LINE_WIDTH, 2);
//---Set up the data series(buffers as series arrays)
   ArraySetAsSeries(openBuffer, true);
   ArraySetAsSeries(highBuffer, true);
   ArraySetAsSeries(lowBuffer, true);
   ArraySetAsSeries(closeBuffer, true);

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---
   int start = MathMax(prev_calculated-1, 0);  //start from the most recent data
   double price =close[rates_total-1]; // starting price
   MathSrand(GetTickCount()); //initialize random seed
//---Generate synthetic data for thechart
   for(int i = start; i < rates_total; i++)
     {
      double change = (MathRand()/ 32768.0)* 0.0002 - 0.0002; //Random price change
      price += change ;    // Update price with the random change
      openBuffer[i]= price;
      highBuffer[i]= price + 0.0002;  //simulated high
      lowBuffer[i]= price - 0.0002; //simulated low
      closeBuffer[i]= price;

     }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
