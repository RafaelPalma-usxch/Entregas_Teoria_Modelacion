run('Parametros_estanques_0.m')

modelo = 'Simulacion_estanques_0';
load_system(modelo)

H0 = 3.00;
nIter = 51;
Hvals = H0 + (0:nIter-1) * 0.01;

tiempoEstanque_01 = NaN(size(Hvals));
tiempoEstanque_02 = NaN(size(Hvals));

for k = 1:numel(Hvals)
    H = Hvals(k); %#ok<NASGU>

    out = sim(modelo, 'ReturnWorkspaceOutputs', 'on');

    sig1 = out.Estanque_01;
    sig2 = out.Estanque_02;

    t1 = sig1.Time;
    y1 = sig1.Data;

    t2 = sig2.Time;
    y2 = sig2.Data;

    idx1 = find(y1(:) >= Hvals(k), 1, 'first');
    idx2 = find(y2(:) >= Hvals(k), 1, 'first');

    if ~isempty(idx1)
        tiempoEstanque_01(k) = t1(idx1);
    end

    if ~isempty(idx2)
        tiempoEstanque_02(k) = t2(idx2);
    end
end

resultado = table(Hvals(:), tiempoEstanque_01(:), tiempoEstanque_02(:), ...
    'VariableNames', {'H', 'TiempoEstanque_01', 'TiempoEstanque_02'});

disp(resultado)

writetable(resultado, 'resultado_estanques.csv')

function [t, y] = getSignalData(sig)
if isa(sig, 'timeseries')
    t = sig.Time;
    y = sig.Data;
elseif isstruct(sig) && isfield(sig, 'time') && isfield(sig, 'signals')
    t = sig.time;
    y = sig.signals.values;
else
    error('Formato de salida no soportado. Usa Timeseries o Structure with time en To Workspace.');
end
end
