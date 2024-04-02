$(document).ready(function() {
    $('#predictionForm').submit(function(event) {
        event.preventDefault(); // Verhindert das Standardverhalten des Formulars

        // Formulardaten sammeln
        var formData = {
            'Ausgangsjahr': $('#ausgangsjahr').val(),
            'Einwohner': $('#einwohner').val(),
            'Häufigkeitszahl': $('#häufigkeitszahl').val()
        };

        // AJAX-Anfrage an die Flask-Route senden
        $.ajax({
            type: 'POST',
            url: '/train_model',
            data: formData,
            dataType: 'json',
            success: function(data) {
                // Vorhersageergebnis anzeigen
                $('#predictionValue').text('Vorhersage: ' + data.prediction);
            }
        });
    });
});
