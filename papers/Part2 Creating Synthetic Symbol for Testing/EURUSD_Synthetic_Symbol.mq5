//+------------------------------------------------------------------+
//|                                       EURUSD_Sythetic Symbol.mq5 |
//|                                   Copyright 2024, MetaQuotes Ltd.|
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#define SYNTHETIC_CSV_FILE_NAME "EURUSD_3_years_synthetic.csv"

// Function to read synthetic data from a CSV file
bool ReadSyntheticDataFromCSV(const string fileName, MqlRates &rates[])
{
    int fileHandle = FileOpen(fileName, FILE_CSV | FILE_READ | FILE_ANSI);
    if (fileHandle == INVALID_HANDLE)
    {
        Print("Error opening file: ", GetLastError());
        return false;
    }

    ArrayResize(rates, 0);

    while (!FileIsEnding(fileHandle))
    {
        string line = FileReadString(fileHandle);
        StringReplace(line, ",", ".");
        string fields[];
        int fieldCount = StringSplit(line, ';', fields);

        if (fieldCount >= 6)
        {
            MqlRates rate;
            rate.time = (datetime)StringToTime(fields[0]);
            rate.open = StringToDouble(fields[1]);
            rate.high = StringToDouble(fields[2]);
            rate.low = StringToDouble(fields[3]);
            rate.close = StringToDouble(fields[4]);
            rate.tick_volume = (long)StringToInteger(fields[5]);
            rate.spread = 0;
            rate.real_volume = 0;

            int currentSize = ArraySize(rates);
            ArrayResize(rates, currentSize + 1);
            rates[currentSize] = rate;
        }
    }

    FileClose(fileHandle);
    Print("Synthetic data successfully read from CSV.");
    return true;
}

void OnStart()
{
   string syntheticSymbol = "SYNTH_EURUSD";

   // Step 1: Create or Reset the Synthetic Symbol
   if (!CustomSymbolCreate(syntheticSymbol))
   {
      Print("Error creating synthetic symbol: ", GetLastError());
      return;
   }

   // Step 2: Configure the Symbol Properties
   CustomSymbolSetInteger(syntheticSymbol, SYMBOL_DIGITS, 5);
   CustomSymbolSetDouble(syntheticSymbol, SYMBOL_POINT, 0.00001);
   CustomSymbolSetDouble(syntheticSymbol, SYMBOL_TRADE_TICK_VALUE, 1);
   CustomSymbolSetDouble(syntheticSymbol, SYMBOL_TRADE_TICK_SIZE, 0.00001);
   CustomSymbolSetInteger(syntheticSymbol, SYMBOL_SPREAD_FLOAT, true);

   // Step 3: Load Data from CSV
   MqlRates rates[];
   if (!ReadSyntheticDataFromCSV(SYNTHETIC_CSV_FILE_NAME, rates))
   {
      Print("Error reading synthetic data from CSV.");
      return;
   }

   // Step 4: Update Rates and Add to Market Watch
   if (!CustomRatesUpdate(syntheticSymbol, rates))
   {
      Print("Error updating synthetic symbol rates: ", GetLastError());
      return;
   }

   MarketBookAdd(syntheticSymbol); // Add to Market Watch
   Print("Synthetic symbol successfully created and added to Market Watch.");
}