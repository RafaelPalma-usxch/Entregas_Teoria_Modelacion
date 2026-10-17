run('Parametros_estanques_0.m')												%Inicializa el archivo con los parametros del sistema.

modelo = 'Simulacion_estanques_0';											%Carga el sistema creado en Simulink, que a su vez usa los parametros del archivo anterior
load_system(modelo)

H0 = 3.00;																	%Define el limite maximo del sistema de llenado, y marca las 50 iteraciones de entrenamiento,
nIter = 51;																	%aumentanto en intervalos de 0.01.
Hvals = H0 + (0:nIter-1) * 0.01;

tiempoEstanque_01 = NaN(size(Hvals));										%Variables donde se asignara el tiempo de respuesta del sistema en llegar al limite de llenado
tiempoEstanque_02 = NaN(size(Hvals));										%en cada iteracion.

for k = 1:numel(Hvals)														%Ciclo que ejecuta las 50 iteraciones del modelo en Simulink.
    H = Hvals(k); %#ok<NASGU>											

    out = sim(modelo, 'ReturnWorkspaceOutputs', 'on');						%Utiliza el bloque "To Workspace" de cada estanque y almacena los valores de salida en la simulacion

    sig1 = out.Estanque_01;
    sig2 = out.Estanque_02;

    t1 = sig1.Time;															%Las variables guardan cada momento de tiempo y su respuesta en metros.
    y1 = sig1.Data;

    t2 = sig2.Time;
    y2 = sig2.Data;

    idx1 = find(y1(:) >= Hvals(k), 1, 'first');								%Compara las salidas de altura hasta que el sistema alcanza el nivel requerido en cada iteracion,
    idx2 = find(y2(:) >= Hvals(k), 1, 'first');								%siendo esta previamente modificada por el controlador PID, para que no tenga sobresaltos,
																			%Garantizando que el sistema se estabiliza una vez llega por primera vez al nivel.
    if ~isempty(idx1)
        tiempoEstanque_01(k) = t1(idx1);									%Se realizan las comparaciones para detener la simulacion cuando se alcanza el valor deseado.
    end

    if ~isempty(idx2)
        tiempoEstanque_02(k) = t2(idx2);
    end
end

resultado = table(Hvals(:), tiempoEstanque_01(:), tiempoEstanque_02(:), ... %Se escribe en una variable "resultado" la altura del ciclo, y los tiempos que se demora cada estanque.
    'VariableNames', {'H', 'TiempoEstanque_01', 'TiempoEstanque_02'});

disp(resultado)

writetable(resultado, 'resultado_estanques.csv')							%El resultado de cada iteracion se escribe un csv.

function [t, y] = getSignalData(sig)										%Funcion que comprueba que las variables se guarden en el formato correcto.
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
