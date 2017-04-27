function varargout = predict(model, data, varargin)
%PREDICT Computes the k-step ahead prediction.
%
%  YP = PREDICT(MODEL, DATA, K)
%     predicts the output of an identified model MODEL K time instants
%     ahead using input-output data history from DATA. The predicted
%     response is computed for the time span covered by DATA.
%
%     MODEL is an identified, linear or nonlinear model, such as IDSS or
%     IDNLARX. If model is originally unavailable, you can estimate one
%     using commands such as AR, ARMAX, TFEST, NLARX on DATA.
%
%     DATA is an IDDATA object containing a record of measured input and
%     output values. If MODEL is a time series model (no input signals),
%     DATA must be specified as an IDDATA object with no inputs or a double
%     matrix of past (already observed) time series values.
%
%     K is the prediction horizon, a positive integer denoting a multiple
%     of data sample time. Old outputs up to time t-K are used to predict
%     the output at time instant t. All relevant inputs (times t, t-1, t-2,
%     ..) are used. K = Inf gives a pure simulation of the system. (Default
%     K = 1).
%
%     The output argument YP is the predicted output returned as an IDDATA
%     object. If DATA contains multiple experiments, so will YP. The time
%     span of output values in YP is same as that of the observed data set
%     DATA. 
%
%  YP = PREDICT(MODEL, DATA, K, OPTIONS)
%     specifies options affecting handling of initial conditions, data
%     offsets and other options controlling the prediction algorithm. See
%     predictOptions for more information.
%
% [YP, X0E, MPRED] = PREDICT(MODEL, DATA,...)
%     returns the estimated values of initial states X0E and a predictor
%     system MPRED. MPRED is a dynamic system whose simulation using
%     [DATA.OutputData, Data.InputData] as input signal yields
%     YP.OutputData as the response (using initial states X0E for state
%     space models). For discrete-time data (time domain data or frequency
%     domain data with Ts>0), MPRED is a discrete-time system, even if
%     MODEL is continuous-time. If DATA is multiexperiment, MPRED is an
%     array of Ne systems, where Ne = number of data experiments.
%
%     When MODEL is a nonlinear ARX (IDNLARX) or a Hammerstein-Wiener
%     (IDNLHW) model, X0 corresponds to an appropriate state definition of
%     the system. See help on IDNLARX and IDNLHW for more information.
%     MPRED is [] for nonlinear systems. For IDNLHW and IDNLGREY models,
%     the predicted response is the same as the simulated response computed
%     using Data.InputData as the input signal.
%
%  When called with no output arguments, PREDICT(...) shows a plot of the
%  predicted response.
%
%  Difference between PREDICT and FORECAST: The PREDICT command predicts
%  response over the time span of DATA, while the FORECAST command performs
%  prediction into unseen future, which is a time range beyond the last
%  time value in DATA. PREDICT is thus a tool for validating the quality of
%  an estimated model (does the prediction result match observed response
%  in DATA.OutputData?). Use PREDICT to ascertain the prediction capability
%  of MODEL before using it for forecasting. The FORECAST command does not
%  apply to nonlinear systems.
%
% See also PREDICTOPTIONS, IDMODEL/FORECAST, PE, IDMODEL/SIM, LSIM,
% COMPARE, IDDATA, IDPAR.

%  Author(s): Rajiv Singh
%  Copyright 2010-2015 The MathWorks, Inc.

narginchk(2,Inf)
modelname = inputname(1);

% check if data/model order is reversed
if isa(data,'DynamicSystem') && (isnumeric(model) || isa(model,'frd') || isa(model,'iddata'))
   datat = data; data = model; model = datat;
   modelname = inputname(2);
end

if isa(model,'FRDModel')
   ctrlMsgUtils.error('Ident:general:InvalidSyntax','predict','predict')
elseif ~isa(model,'idlti') && ~isa(model,'idnlmodel')
   try
      model = idpack.ltidata.convertToIdlti(model,'predict');
   catch 
      ctrlMsgUtils.error('Ident:general:unsupportedModelType',class(model),'predict');
   end
end

if isempty(model.Name), model.Name = modelname; end

no = nargout;
Warn = ctrlMsgUtils.SuspendWarnings('Ident:transformation:InternalDelaysAreStates');
try
   [yp, varargout{2:no}] = predict_(model, data, varargin{:});
catch E
   throw(E)
end
delete(Warn)

if no==0
   model = copyDataMetaData(model, yp, true);
   utidplot(model,yp,'Predicted')
else
   if isnumeric(data)
      yp = pvget(yp,'OutputData');
      if isscalar(yp), yp = yp{1}; end
   end
   varargout{1} = yp;
end
